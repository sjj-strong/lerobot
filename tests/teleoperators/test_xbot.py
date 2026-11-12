# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
import time

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.hardware


@pytest.fixture
def xbot_calibration():
    return {
        "version": 1,
        "backend": "evdev",
        "device": {"name": "XBot test", "vendor": 1, "product": 2},
        "axes": {
            name: {"code": code, "rest": 0, "positive": 100, "min": -100, "max": 100}
            for code, name in enumerate(("lx", "ly", "rx", "ry", "lt", "rt", "yaw"))
        },
        "buttons": {
            name: 300 + index for index, name in enumerate(("rb", "lb", "a", "x", "menu", "y", "b", "view"))
        },
    }


def test_xbot_maps_native_axis_ranges_and_rejects_invalid_layout(xbot_calibration):
    from copy import deepcopy

    from lerobot.teleoperators.xbot.input import XBotMapping

    mapping = XBotMapping(xbot_calibration)
    axes, buttons = mapping.decode(dict.fromkeys(range(7), 0), set())
    assert all(value == 0 for value in axes.values())
    axes, buttons = mapping.decode({**dict.fromkeys(range(7), 0), 0: 54, 4: -100}, {300})
    assert axes["lx"] == pytest.approx(0.5)
    assert axes["lt"] == 0
    assert buttons["rb"]
    for mutation in ("duplicate", "range", "nan", "backend"):
        invalid = deepcopy(xbot_calibration)
        if mutation == "duplicate":
            invalid["axes"]["ly"]["code"] = 0
        elif mutation == "range":
            invalid["axes"]["lx"]["positive"] = 0
        elif mutation == "nan":
            invalid["axes"]["lx"]["rest"] = float("nan")
        else:
            invalid["backend"] = "ros_joy"
        with pytest.raises(ValueError):
            XBotMapping(invalid)
    with pytest.raises(ValueError):
        mapping.decode({**dict.fromkeys(range(7), 0), 0: float("nan")}, set())
    with pytest.raises(ValueError):
        mapping.decode({}, set())


def _xbot_reading(**overrides):
    reading = {
        "axes": dict.fromkeys(("lx", "ly", "rx", "ry", "lt", "rt", "yaw"), 0.0),
        "buttons": dict.fromkeys(("rb", "lb", "a", "x", "menu", "y", "b", "view"), False),
        "received_at": time.monotonic(),
    }
    for name, value in overrides.items():
        if name in reading["axes"]:
            reading["axes"][name] = value
        elif name in reading["buttons"]:
            reading["buttons"][name] = value
        else:
            reading[name] = value
    return reading


def test_xbot_requires_release_then_outputs_normalized_increment(xbot_calibration):
    from lerobot.teleoperators.xbot.input import XBotInput

    device = XBotInput("unused", xbot_calibration)
    context = {
        "tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1.0]),
        "gripper_open": True,
        "action_scale": [0.02, 0.1, 1],
    }
    device.on_reset(context)
    assert not device.action(_xbot_reading(rb=True, ly=1), context).driving
    device.action(_xbot_reading(), context)
    moved = device.action(_xbot_reading(rb=True, ly=1), context)
    assert moved.driving
    np.testing.assert_array_equal(moved.parts["arm"], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(moved.info["xbot_delta"], [1, 0, 0, 0, 0, 0])
    combined = device.action(_xbot_reading(rb=True, ly=1, rt=1, ry=1, yaw=1), context)
    np.testing.assert_array_equal(combined.parts["arm"], [1, 0, 1, 1, 0, 1])
    context["tcp_pose"][0] = 0.1
    held = device.action(_xbot_reading(), context)
    np.testing.assert_array_equal(held.parts["arm"], np.zeros(6))
    np.testing.assert_array_equal(held.info["xbot_delta"], np.zeros(6))
    slow = device.action(_xbot_reading(rb=True, lb=True, ly=1), context)
    np.testing.assert_array_equal(slow.parts["arm"], [device.precision_scale, 0, 0, 0, 0, 0])
    assert slow.parts["end_effector"][0] == 1
    np.testing.assert_array_equal(slow.info["xbot_delta"], [device.precision_scale, 0, 0, 0, 0, 0])
    context["tcp_pose"][0] = 0.2
    centered = device.action(_xbot_reading(rb=True), context)
    assert centered.driving
    np.testing.assert_array_equal(centered.info["xbot_delta"], np.zeros(6))


def test_xbot_tcp_frame_and_toggle_cycle(xbot_calibration):
    from lerobot.teleoperators.xbot.input import XBotInput

    device = XBotInput("unused", xbot_calibration)
    rotation = Rotation.from_euler("z", 90, degrees=True)
    context = {
        "tcp_pose": np.r_[np.zeros(3), rotation.as_quat()],
        "gripper_open": True,
        "action_scale": [0.02, 0.1, 1],
    }
    device.on_reset(context)
    device.action(_xbot_reading(), context)
    toggle = device.action(_xbot_reading(rb=True, x=True, ly=1), context)
    np.testing.assert_allclose(toggle.parts["arm"][:3], 0)
    np.testing.assert_array_equal(toggle.info["xbot_delta"], np.zeros(6))
    moved = device.action(_xbot_reading(rb=True, ly=1), context)
    assert moved.parts["arm"][1] == pytest.approx(1)
    np.testing.assert_allclose(moved.info["xbot_delta"], [0, 1, 0, 0, 0, 0], atol=1e-12)


def test_xbot_invalid_axes_disarm(xbot_calibration):
    from lerobot.teleoperators.xbot.input import XBotInput

    device = XBotInput("unused", xbot_calibration)
    context = {
        "tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1.0]),
        "gripper_open": True,
        "action_scale": [0.02, 0.1, 1],
    }
    device.on_reset(context)
    device.action(_xbot_reading(), context)
    device.action(_xbot_reading(rb=True), context)
    with pytest.raises(ValueError):
        device.action(_xbot_reading(rb=True, lx=float("nan")), context)
    assert not device.action(_xbot_reading(rb=True, ly=1), context).driving
    device.action(_xbot_reading(), context)
    assert device.action(_xbot_reading(rb=True, ly=1), context).driving
    with pytest.raises(ValueError):
        device.action(_xbot_reading(rb=True, lx=float("nan")), context)


def test_xbot_accepts_old_sample_without_age_limit(xbot_calibration):
    from lerobot.teleoperators.xbot.input import XBotInput

    device = XBotInput("unused", xbot_calibration)
    context = {
        "tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1.0]),
        "gripper_open": True,
        "action_scale": [0.02, 0.1, 1],
    }
    result = device.action(_xbot_reading(received_at=time.monotonic() - 2), context)
    assert result.info["xbot_input_age"] >= 2


def test_xbot_events_are_edges_and_finalize_fires_once(xbot_calibration):
    from lerobot.teleoperators.xbot.input import XBotInput

    device = XBotInput("unused", xbot_calibration, view_hold_s=0.01)
    context = {
        "tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1.0]),
        "gripper_open": True,
        "action_scale": [0.02, 0.1, 1],
    }
    device.on_reset(context)
    device.action(_xbot_reading(), context)
    first = device.action(_xbot_reading(rb=True, a=True, menu=True), context)
    assert first.parts["end_effector"][0] == -1
    assert first.info["xbot_events"] == ["start"]
    held = device.action(_xbot_reading(rb=True, a=True, menu=True), context)
    assert held.parts["end_effector"][0] == -1
    assert held.info["xbot_events"] == []
    device.action(_xbot_reading(view=True), context)
    time.sleep(0.012)
    assert device.action(_xbot_reading(view=True), context).info["xbot_events"] == ["finalize"]
    assert device.action(_xbot_reading(view=True), context).info["xbot_events"] == []
