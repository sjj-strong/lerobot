# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""LeRobot UR adapter with optional cameras and a serial Robotiq gripper."""

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _scipy_available

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .config_ur import URConfig, vector
from .controller import RTDEController

JOINT_NAMES = tuple(f"joint_{i}.pos" for i in range(1, 7))
TCP_NAMES = tuple(f"tcp.{s}" for s in ("x", "y", "z", "rx", "ry", "rz"))
WRENCH_NAMES = tuple(f"wrench.{s}" for s in ("fx", "fy", "fz", "tx", "ty", "tz"))


class UR(Robot):
    config_class = URConfig
    name = "ur"
    supports_command_hold = True

    def __init__(self, config: URConfig) -> None:
        super().__init__(config)
        self.config = config
        self.controller = RTDEController(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.gripper = None
        self._connected = False
        self._hold_pose = None
        self._active_axes = np.zeros(6, dtype=bool)

    @property
    def action_features(self) -> dict:
        names = JOINT_NAMES if self.config.mode.startswith("joint") else TCP_NAMES
        if self.config.mode.endswith("velocity"):
            names = tuple(
                n.replace(".pos", ".vel") if n.startswith("joint") else n.replace("tcp.", "tcp_velocity.")
                for n in names
            )
        # Force mode commands a pose equilibrium, never a measured wrench.
        if self.config.mode == "cartesian_force":
            names = tuple(n.replace("tcp.", "force_target.") for n in names)
        return dict.fromkeys(names, float) | ({"gripper.width": float} if self.config.gripper else {})

    @property
    def observation_features(self) -> dict:
        result = dict.fromkeys(
            JOINT_NAMES + TCP_NAMES + (WRENCH_NAMES if self.config.include_wrench else ()), float
        )
        if self.config.gripper:
            result["gripper.width"] = float
        return result | {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """UR calibration is owned by the robot controller."""

    def configure(self) -> None:
        """RTDE settings are applied once on connection."""

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("UR is already connected")
        try:
            self.controller.connect()
            if self.config.gripper is not None:
                from .robotiq import RobotiqGripper

                self.gripper = RobotiqGripper(**asdict(self.config.gripper))
                self.gripper.connect()
            for camera in self.cameras.values():
                camera.connect()
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise RuntimeError("Expected connected UR")
        state = self.controller.state()
        result = dict(zip(JOINT_NAMES, state["joint"], strict=True)) | dict(
            zip(TCP_NAMES, state["tcp"], strict=True)
        )
        if self.config.include_wrench:
            result.update(zip(WRENCH_NAMES, state["wrench"], strict=True))
        if self.gripper is not None:
            result["gripper.width"] = self.gripper.position
        result.update({name: cam.async_read() for name, cam in self.cameras.items()})
        return result

    def get_diagnostics(self) -> dict:
        """Return display-only counters and feedback, separate from observations."""
        return self.controller.get_diagnostics()

    def send_action(self, action: dict) -> dict:
        if not self.is_connected:
            raise RuntimeError("Expected connected UR")
        if set(action) != set(self.action_features):
            raise ValueError(f"Expected action keys {list(self.action_features)}, found {list(action)}")
        names = list(self.action_features)[:6]
        target = vector([action[key] for key in names], "action")
        width = None
        if self.gripper is not None:
            width = float(action["gripper.width"])
            if not np.isfinite(width):
                raise ValueError("Expected finite gripper width")
            width = float(np.clip(width, 0, self.config.gripper.max_width))
        accepted = self.controller.send(target)
        result = dict(zip(names, accepted, strict=True))
        if width is not None:
            try:
                self.gripper.move(width)
            except BaseException:
                self.stop()
                raise
            result["gripper.width"] = width
        return result

    def prepare_pose_target(self, delta: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Latch released axes at feedback and retain idle-axis equilibrium."""
        delta, pose = vector(delta, "pose increment"), vector(pose, "pose feedback")
        actual_rotation = Rotation.from_rotvec(pose[3:])
        desired_rotation = Rotation.from_euler("xyz", delta[3:]) * actual_rotation
        offset = np.r_[delta[:3], (desired_rotation * actual_rotation.inv()).as_rotvec()]
        active = np.abs(offset) > 1e-12
        if self._hold_pose is None:
            self._hold_pose = pose.copy()
        released = self._active_axes & ~active
        hold = self._hold_pose.copy()
        hold[:3][released[:3]] = pose[:3][released[:3]]
        held_rotation = Rotation.from_rotvec(hold[3:])
        release_error = (actual_rotation * held_rotation.inv()).as_rotvec()
        held_rotation = Rotation.from_rotvec(release_error * released[3:]) * held_rotation
        hold[3:] = held_rotation.as_rotvec()
        target_position = np.where(active[:3], pose[:3] + delta[:3], hold[:3])
        error = (desired_rotation * held_rotation.inv()).as_rotvec()
        target_rotation = Rotation.from_rotvec(error * active[3:]) * held_rotation
        self._hold_pose, self._active_axes = hold, active
        return np.r_[target_position, target_rotation.as_rotvec()]

    def hold(self) -> None:
        self.stop()

    def stop(self) -> None:
        self.controller.stop()
        self._hold_pose = None
        self._active_axes[:] = False

    def move_joints(self, target: np.ndarray) -> None:
        self.controller.move(target, joint=True)

    def move_pose(self, target: np.ndarray) -> None:
        self.controller.move(target)

    def disconnect(self) -> None:
        self._hold_pose = None
        self._active_axes[:] = False
        self.controller.disconnect()
        try:
            if self.gripper is not None:
                self.gripper.disconnect()
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
        finally:
            self._connected = False
