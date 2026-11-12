#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""UR teleoperation using the editable config.py.

Host, from /media/robot/E48F044ACB22FD69/sjj/Rlinf:
    /media/robot/E48F044ACB22FD69/sjj/Rlinf/.venv-ur/bin/python external/lerobot/examples/ur/teleoperate.py
    Requires Python >=3.12 with this LeRobot checkout installed.
Inside the container, from /workspace/RLinf (independent LeRobot environment):
    /opt/venv/lerobot/bin/python external/lerobot/examples/ur/teleoperate.py
    /opt/venv/gr00t/bin/python is Python 3.11 and cannot run LeRobot 0.7.1.
"""

import logging

import config
from session import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(config, record=False)
