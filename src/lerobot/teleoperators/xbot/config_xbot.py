# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xbot")
@dataclass(kw_only=True)
class XBotConfig(TeleoperatorConfig):
    device: str
    calibration_path: str
    deadzone: float = 0.08
    precision_scale: float = 0.5
    view_hold_s: float = 0.5
    mapping_profile: str = "standard"
