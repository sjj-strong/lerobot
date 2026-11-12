# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Edit these Python objects before running teleoperate.py or record.py."""

from pathlib import Path

from tasks import Task

from lerobot.robots.ur import URConfig
from lerobot.teleoperators.xbot import XBotConfig

ROBOT = URConfig(
    robot_ip="169.254.138.15",
    mode="cartesian_force",
    cameras={},  # Optional: name -> OpenCVCameraConfig / RealSenseCameraConfig.
    gripper=None,  # Optional: RobotiqConfig(port='/dev/ttyUSB0').
    include_wrench=True,
    payload_mass=None,  # None preserves the UR controller payload setting.
    payload_cog=(0.0, 0.0, 0.0),
)
TELEOP = XBotConfig(
    device="/dev/input/by-id/usb-Zikway_HID_gamepad-event-joystick",
    calibration_path="xbot_zikway.json",
    mapping_profile="standard",
)
TASK = Task()
# Example replacement: PegInsertionTask(home_joints=(q1, q2, q3, q4, q5, q6)).
# ROBOT.mode selects the control method; force.damping=None derives 2*sqrt(K).
ENABLE_MOTION = False
FPS = 30
TRANSLATION = 0.002  # m/frame in position/force modes; m/s in velocity mode.
ROTATION = 0.01  # rad/frame in position/force modes; rad/s in velocity mode.
JOINT_INDEX = 0  # 0..5, left-stick Y controls the selected joint.
JOINT_STEP = 0.005  # rad/frame or rad/s according to mode.
DATASET_ROOT = Path("dataset/lerobot/ur")
REPO_ID = "local/ur"
RESUME = False
NUM_EPISODES = 20  # Number of new successful episodes in this invocation.
USE_VIDEOS = True
