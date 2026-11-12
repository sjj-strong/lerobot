# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Single RTDE command owner with a command-age watchdog and explicit stops."""

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

from .config_ur import ForceConfig, URConfig, vector


def limit_target(config: URConfig, target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    target, actual = vector(target, "target"), vector(actual, "actual")
    if config.mode == "joint_position":
        return actual + np.clip(target - actual, -config.joint_step, config.joint_step)
    if config.mode.endswith("velocity"):
        bounds = (
            np.full(6, config.joint_speed)
            if config.mode.startswith("joint")
            else np.r_[np.full(3, config.linear_speed), np.full(3, config.angular_speed)]
        )
        return np.clip(target, -bounds, bounds)
    rotation = Rotation.from_rotvec(actual[3:])
    delta = (Rotation.from_rotvec(target[3:]) * rotation.inv()).as_rotvec()
    delta *= min(1, config.rotation_step / max(np.linalg.norm(delta), 1e-12))
    return np.r_[
        actual[:3] + np.clip(target[:3] - actual[:3], -config.translation_step, config.translation_step),
        (Rotation.from_rotvec(delta) * rotation).as_rotvec(),
    ]


def force_wrench(
    config: ForceConfig, target: np.ndarray, pose: np.ndarray, velocity: np.ndarray
) -> np.ndarray:
    to_task = Rotation.from_rotvec(np.array(config.task_frame[3:])).inv()
    rotation = Rotation.from_rotvec(pose[3:])
    error = np.r_[
        to_task.apply(target[:3] - pose[:3]),
        to_task.apply((Rotation.from_rotvec(target[3:]) * rotation.inv()).as_rotvec()),
    ]
    speed = np.r_[to_task.apply(velocity[:3]), to_task.apply(velocity[3:])]
    tcp_to_task = to_task * rotation
    feedforward = (
        np.asarray(config.wrench)
        + np.r_[tcp_to_task.apply(config.tcp_wrench[:3]), tcp_to_task.apply(config.tcp_wrench[3:])]
    )
    wrench = feedforward + np.asarray(config.stiffness) * error - config.resolved_damping * speed
    return np.clip(wrench, -np.asarray(config.max_wrench), config.max_wrench) * config.selection_vector


class RTDEController:
    """Own SDK connections; only the control thread writes while streaming."""

    def __init__(self, config: URConfig) -> None:
        require_package("scipy", "ur")
        self.config = config
        self.control = self.receive = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._target = None
        self._stamp = 0.0
        self._stop_error = None
        self._error = None
        self._receive_lock = threading.Lock()
        self._feedback_stamp = None
        self._feedback_time = 0.0
        self._target_count = self._control_count = 0
        self._robot_mode = self._safety_mode = None
        self._wrench = None
        self._wrench_time = None

    def connect(self) -> None:
        if self.control is not None or self.receive is not None:
            raise RuntimeError("Disconnect the existing UR connection before reconnecting")
        self._error = self._stop_error = None
        with self._lock:
            self._target_count = self._control_count = 0
            self._robot_mode = self._safety_mode = None
            self._wrench = self._wrench_time = None
        self._feedback_stamp = None
        self._feedback_time = 0.0
        import rtde_control
        import rtde_receive

        try:
            self.receive = rtde_receive.RTDEReceiveInterface(self.config.robot_ip, self.config.frequency)
            self.control = rtde_control.RTDEControlInterface(self.config.robot_ip, self.config.frequency)
            if self.config.payload_mass is not None:
                self._ok(self.control.setPayload(self.config.payload_mass, list(self.config.payload_cog)))
            if self.config.mode == "cartesian_force":
                self._ok(self.control.forceModeSetDamping(self.config.force.force_damping))
                self._ok(self.control.forceModeSetGainScaling(self.config.force.gain_scaling))
                if self.config.force.zero_ft_sensor:
                    self._ok(self.control.zeroFtSensor())
            self.state()
        except BaseException:
            self.disconnect()
            raise

    @staticmethod
    def _ok(result):
        if result is False:
            raise RuntimeError("UR controller rejected RTDE command")

    def state(self) -> dict[str, np.ndarray]:
        if self.receive is None or not self.receive.isConnected():
            raise RuntimeError("Expected connected UR RTDE receiver")
        if self._error is not None:
            raise RuntimeError("UR control loop failed; reconnect before continuing") from self._error
        with self._receive_lock:
            robot_mode, safety_mode = self.receive.getRobotMode(), self.receive.getSafetyMode()
            with self._lock:
                self._robot_mode, self._safety_mode = robot_mode, safety_mode
            if self.receive.isProtectiveStopped() or self.receive.isEmergencyStopped():
                raise RuntimeError("UR safety stop is active")
            now, stamp = time.monotonic(), float(self.receive.getTimestamp())
            if not np.isfinite(stamp) or (self._feedback_stamp is not None and stamp < self._feedback_stamp):
                raise ValueError("Expected finite monotonic UR feedback timestamp")
            if self._feedback_stamp is None or stamp > self._feedback_stamp:
                self._feedback_stamp, self._feedback_time = stamp, now
            elif now - self._feedback_time > self.config.command_timeout:
                raise TimeoutError("UR feedback timestamp stopped advancing")
            if safety_mode not in (1, 2) or robot_mode != 7:
                raise RuntimeError("Expected running UR in normal or reduced safety mode")
            result = {
                "joint": vector(self.receive.getActualQ(), "actual joints"),
                "tcp": vector(self.receive.getActualTCPPose(), "actual TCP"),
                "velocity": vector(self.receive.getActualTCPSpeed(), "actual TCP speed"),
                "wrench": vector(self.receive.getActualTCPForce(), "actual wrench"),
            }
            with self._lock:
                self._wrench = result["wrench"].copy()
                self._wrench_time = self._feedback_time
            return result

    def send(self, target: np.ndarray) -> np.ndarray:
        state = self.state()
        actual = state["joint" if self.config.mode.startswith("joint") else "tcp"]
        accepted = limit_target(self.config, target, actual)
        with self._lock:
            self._target, self._stamp = accepted.copy(), time.monotonic()
            self._target_count += 1
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="ur-rtde", daemon=True)
            self._thread.start()
        return accepted

    def get_diagnostics(self) -> dict:
        """Copy counters and last feedback without SDK reads or dataset fields."""
        with self._lock:
            now = time.monotonic()
            return {
                "sampled_at": now,
                "target_count": self._target_count,
                "control_count": self._control_count,
                "robot_mode": self._robot_mode,
                "safety_mode": self._safety_mode,
                "wrench": None if self._wrench is None else self._wrench.copy(),
                "feedback_age_s": None if self._wrench_time is None else now - self._wrench_time,
                "error": str(self._error) if self._error is not None else None,
            }

    def _write(self, target, state):
        cfg, control, dt = self.config, self.control, 1 / self.config.frequency
        if cfg.mode == "joint_position":
            return control.servoJ(target.tolist(), 0, 0, dt, cfg.lookahead, cfg.servo_gain)
        if cfg.mode == "cartesian_position":
            return control.servoL(target.tolist(), 0, 0, dt, cfg.lookahead, cfg.servo_gain)
        if cfg.mode == "joint_velocity":
            return control.speedJ(target.tolist(), cfg.acceleration, dt)
        if cfg.mode == "cartesian_velocity":
            return control.speedL(target.tolist(), cfg.acceleration, dt)
        f = cfg.force
        return control.forceMode(
            list(f.task_frame),
            list(f.selection_vector),
            force_wrench(f, target, state["tcp"], state["velocity"]).tolist(),
            2,
            list(f.limits),
        )

    def _run(self):
        try:
            while not self._stop.is_set():
                tick = self.control.initPeriod()
                with self._lock:
                    target, stamp = self._target.copy(), self._stamp
                if time.monotonic() - stamp > self.config.command_timeout:
                    raise TimeoutError("UR command watchdog expired")
                state = self.state()
                if self._stop.is_set():
                    break
                if time.monotonic() - stamp > self.config.command_timeout:
                    raise TimeoutError("UR command expired while reading feedback")
                self._ok(self._write(target, state))
                with self._lock:
                    self._control_count += 1
                self.control.waitPeriod(tick)
        except Exception as error:
            self._error = error
        finally:
            try:
                self._stop_sdk()
            except Exception as error:
                self._stop_error = error
                self._error = error

    def _stop_sdk(self):
        if self.config.mode == "cartesian_force":
            try:
                self._ok(self.control.forceModeStop())
            finally:
                self._ok(self.control.stopL(self.config.acceleration))
        elif self.config.mode.endswith("velocity"):
            self._ok(self.control.speedStop(self.config.acceleration))
        else:
            self._ok(self.control.servoStop(self.config.acceleration))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            deadline = time.monotonic() + 5
            while self._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.005)
            if self._thread.is_alive():
                raise TimeoutError("UR control owner did not stop; connections retained")
            self._thread = None
        if self._stop_error is not None:
            raise RuntimeError("UR stop failed; connections retained for recovery") from self._stop_error

    def move(self, target: np.ndarray, *, joint: bool = False) -> None:
        self.stop()
        self.state()
        target = vector(target, "reset target").tolist()
        method = self.control.moveJ if joint else self.control.moveL
        try:
            self._ok(method(target, self.config.reset_speed, self.config.reset_acceleration, True))
            while self.control.getAsyncOperationProgress() >= 0:
                self.state()
                time.sleep(0.01)
        finally:
            stop = self.control.stopJ if joint else self.control.stopL
            self._ok(stop(self.config.reset_acceleration))

    def disconnect(self) -> None:
        self.stop()
        for name in ("control", "receive"):
            device = getattr(self, name)
            if device is not None:
                device.disconnect()
                setattr(self, name, None)
