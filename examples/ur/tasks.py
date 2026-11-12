# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Task-specific reset hooks; hardware motion primitives belong to UR."""

from dataclasses import dataclass

from lerobot.robots.ur.config_ur import vector


@dataclass
class Task:
    description: str = "UR teleoperation"

    def reset(self, robot):
        """Generic task stops control and leaves the arm where the operator placed it."""
        robot.stop()


@dataclass
class PegInsertionTask(Task):
    description: str = "Insert the peg into the hole"
    home_joints: tuple | None = None
    retract_distance: float = 0.05

    def __post_init__(self):
        if self.home_joints is None:
            raise ValueError("Peg insertion requires explicitly configured home_joints in radians")
        vector(self.home_joints, "home_joints")
        if not 0 <= self.retract_distance <= 0.3:
            raise ValueError("Expected retract_distance in [0,0.3] metres")

    def reset(self, robot):
        """Retract in base Z, then move to this task's configured home joints."""
        robot.stop()
        pose = robot.controller.state()["tcp"].copy()
        pose[2] += self.retract_distance
        if self.retract_distance:
            robot.move_pose(pose)
        robot.move_joints(self.home_joints)
