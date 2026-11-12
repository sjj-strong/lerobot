# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""UR RTDE configuration in SI units; orientations use rotation vectors."""

from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig

MODES = ("joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity", "cartesian_force")


def vector(value, name, size=6):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values for {name}, found {value!r}")
    return result


@dataclass
class ForceConfig:
    task_frame: tuple = (0, 0, 0, 0, 0, 0)
    selection_vector: tuple = (1, 1, 1, 1, 1, 1)
    wrench: tuple = (0, 0, 0, 0, 0, 0)
    tcp_wrench: tuple = (0, 0, 0, 0, 0, 0)
    stiffness: tuple = (1200, 1200, 1200, 100, 100, 100)
    damping: tuple = (70, 70, 70, 20, 20, 20)
    max_wrench: tuple = (40, 40, 40, 2, 2, 2)
    limits: tuple = (0.2, 0.2, 0.2, 0.05, 0.05, 0.3)
    gain_scaling: float = 1.0
    force_damping: float = 0.3
    zero_ft_sensor: bool = False

    def __post_init__(self):
        for name in (
            "task_frame",
            "selection_vector",
            "wrench",
            "tcp_wrench",
            "stiffness",
            "damping",
            "max_wrench",
            "limits",
        ):
            vector(getattr(self, name), name)
        if not np.isin(self.selection_vector, [0, 1]).all():
            raise ValueError("Expected binary force selection vector")
        for name in ("stiffness", "damping", "max_wrench", "limits"):
            a = vector(getattr(self, name), name)
            if np.any(a < 0) or (name in ("max_wrench", "limits") and np.any(a == 0)):
                raise ValueError(f"Expected positive bounds or nonnegative gains: {name}")
        if not 0 <= self.gain_scaling <= 2 or not 0 <= self.force_damping <= 1:
            raise ValueError("Expected gain scaling in [0,2] and force damping in [0,1]")


@dataclass
class RobotiqConfig:
    port: str
    baudrate: int = 115200
    slave_id: int = 9
    max_width: float = 0.085


@RobotConfig.register_subclass("ur")
@dataclass(kw_only=True)
class URConfig(RobotConfig):
    robot_ip: str
    mode: str = "cartesian_position"
    frequency: float = 500
    command_timeout: float = 0.5
    include_wrench: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gripper: RobotiqConfig | None = None
    payload_mass: float | None = None
    payload_cog: tuple = (0, 0, 0)
    force: ForceConfig = field(default_factory=ForceConfig)
    joint_step: float = 0.008
    translation_step: float = 0.04
    rotation_step: float = 0.1
    joint_speed: float = 0.2
    linear_speed: float = 0.1
    angular_speed: float = 0.3
    acceleration: float = 0.5
    lookahead: float = 0.1
    servo_gain: float = 300
    reset_speed: float = 0.1
    reset_acceleration: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in MODES:
            raise ValueError(f"Expected mode in {MODES}, found {self.mode!r}")
        for name in (
            "frequency",
            "command_timeout",
            "joint_step",
            "translation_step",
            "rotation_step",
            "joint_speed",
            "linear_speed",
            "angular_speed",
            "acceleration",
            "reset_speed",
            "reset_acceleration",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Expected finite positive {name}")
        if self.frequency > 500 or not 0.03 <= self.lookahead <= 0.2 or not 100 <= self.servo_gain <= 2000:
            raise ValueError("Expected frequency <=500, lookahead [0.03,0.2], servo_gain [100,2000]")
        vector(self.payload_cog, "payload_cog", 3)
        if self.payload_mass is not None and (not np.isfinite(self.payload_mass) or self.payload_mass < 0):
            raise ValueError("Expected nonnegative finite payload mass")
