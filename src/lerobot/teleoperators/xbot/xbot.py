# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""XBot Teleoperator; events are separate from normalized six-axis actions."""

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

from lerobot.robots.ur.ur import JOINT_NAMES, TCP_NAMES
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_xbot import XBotConfig
from .input import XBotInput

DELTA_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz")


class XBot(Teleoperator):
    config_class = XBotConfig
    name = "xbot"

    def __init__(self, config: XBotConfig) -> None:
        require_package("scipy", "xbot")
        super().__init__(config)
        self.config = config
        self.input = XBotInput(
            config.device,
            config.calibration_path,
            deadzone=config.deadzone,
            precision_scale=config.precision_scale,
            view_hold_s=config.view_hold_s,
            mapping_profile=config.mapping_profile,
        )
        self._feedback = None
        self.events = []
        self.driving = False
        self.joint_axis = 0.0

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(DELTA_NAMES + ("gripper.open",), float)

    @property
    def feedback_features(self) -> dict:
        return dict.fromkeys(
            TCP_NAMES + ("gripper.open", "rotation_scale", "velocity_mode", "joint_mode"), float
        )

    @property
    def is_connected(self) -> bool:
        return self.input.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Use the version-1 evdev calibration supplied by calibration_path."""

    def configure(self) -> None:
        """Configuration is validated when loading the calibration."""

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("XBot is already connected")
        self.input.connect()

    def reset(self) -> None:
        self.input.on_reset()
        self.driving = False
        self.events = []

    def send_feedback(self, feedback: dict) -> None:
        self._feedback = feedback.copy()

    def get_action(self) -> dict:
        if self._feedback is None:
            raise RuntimeError("XBot requires current robot feedback before get_action")
        feedback = self._feedback
        pose = np.array([feedback[name] for name in TCP_NAMES])
        context = {
            "tcp_pose": np.r_[pose[:3], Rotation.from_rotvec(pose[3:]).as_quat()],
            "gripper_open": feedback.get("gripper.open", True),
            "action_scale": (1.0, feedback.get("rotation_scale", 0.02)),
            "rotation_action_type": "angular_velocity"
            if feedback.get("velocity_mode", False)
            else "euler_xyz",
        }
        if feedback.get("joint_mode", False):
            context["teleop_frame"] = "base"
        try:
            reading = self.input.get_observation()
            result = self.input.action(reading, context)
        except BaseException:
            self.driving = False
            self.events = []
            raise
        self.driving, self.events = result.driving, result.info["xbot_events"]
        self.joint_axis = (
            reading["axes"]["ly"] * (self.config.precision_scale if reading["buttons"]["lb"] else 1.0)
            if self.driving
            else 0.0
        )
        return dict(zip(DELTA_NAMES, result.parts["arm"], strict=True)) | {
            "gripper.open": float(result.parts["end_effector"][0] > 0)
        }

    def disconnect(self) -> None:
        self.input.disconnect()
        self.driving = False
        self._feedback = None


def to_ur_action(
    robot,
    action,
    observation,
    *,
    translation=0.005,
    rotation=0.02,
    joint_axis=0.0,
    joint_index=0,
    joint_step=0.005,
):
    """Convert normalized input to mode-specific SI commands from measured state."""
    mode = robot.config.mode
    delta = np.array([action[name] for name in DELTA_NAMES], dtype=float)
    if not np.isfinite(delta).all() or not np.isfinite([translation, rotation, joint_axis, joint_step]).all():
        raise ValueError("Expected finite teleoperation values")
    if mode.startswith("joint"):
        if not 0 <= joint_index < 6:
            raise ValueError("Expected joint_index in [0,5]")
        target = (
            np.array([observation[name] for name in JOINT_NAMES])
            if mode.endswith("position")
            else np.zeros(6)
        )
        target[joint_index] += joint_axis * joint_step
    elif mode == "cartesian_velocity":
        target = delta * np.r_[np.full(3, translation), np.full(3, rotation)]
    else:
        pose = np.array([observation[name] for name in TCP_NAMES])
        target = robot.prepare_pose_target(delta * np.r_[np.full(3, translation), np.full(3, rotation)], pose)
    result = dict(zip(list(robot.action_features)[:6], target, strict=True))
    if robot.config.gripper:
        result["gripper.width"] = robot.config.gripper.max_width if action["gripper.open"] else 0.0
    return result
