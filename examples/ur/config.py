# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Edit these Python objects before running teleoperate.py or record.py."""

from pathlib import Path

from tasks import Task

from lerobot.robots.ur import ForceConfig, URConfig
from lerobot.teleoperators.xbot import XBotConfig

ROBOT = URConfig(
    robot_ip="169.254.138.15",
    mode="cartesian_force",
    cameras={},  # Optional: name -> OpenCVCameraConfig / RealSenseCameraConfig.
    gripper=None,  # Optional: RobotiqConfig(port='/dev/ttyUSB0').
    include_wrench=True,
    # RLinf realworld_ur_peginsertion_collect_data.yaml arm settings.
    frequency=500.0,
    command_timeout=10.0,
    payload_mass=1.8,
    payload_cog=(0.0, 0.0, 0.066),
    force=ForceConfig(
        task_frame=(0, 0, 0, 0, 0, 0),
        selection_vector=(1, 1, 1, 0, 0, 1),
        wrench=(0, 0, 0, 0, 0, 0),
        tcp_wrench=(0, 0, 0, 0, 0, 0),
        stiffness=(1200, 1200, 1200, 2000, 2000, 2000),
        damping=None,
        max_wrench=(40, 40, 40, 4, 4, 4),
        limits=(0.2, 0.2, 0.2, 0.05, 0.05, 0.3),
        gain_scaling=1.5,
        force_damping=0.15,
        zero_ft_sensor=True,
    ),
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

# Official LeRobot Rerun backend; None/None opens a viewer in this desktop session.
DISPLAY_DATA = True
DISPLAY_FPS = 10
DISPLAY_IP = None
DISPLAY_PORT = None
