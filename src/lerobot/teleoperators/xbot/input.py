# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrated Linux evdev decoding and XBot event state machine."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _scipy_available

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

Features = dict
Observation = dict


@dataclass
class TeleopAction:
    parts: dict
    driving: bool
    info: dict


AXES = ("lx", "ly", "rx", "ry", "lt", "rt", "yaw")
BUTTONS = ("rb", "lb", "a", "x", "menu", "y", "b", "view")


class XBotMapping:
    """Decode versioned evdev calibration into signed axes and buttons.

    Args:
        calibration: Native event codes, raw axis ranges and device identity.
        deadzone: Fraction of normalized travel ignored around neutral.
    """

    def __init__(self, calibration: Mapping[str, Any], deadzone: float = 0.08) -> None:
        self.calibration = copy.deepcopy(dict(calibration))
        self.deadzone = float(deadzone)
        try:
            self._validate()
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("Expected complete native evdev calibration.") from error

    def _validate(self) -> None:
        cfg = self.calibration
        if cfg["version"] != 1 or cfg["backend"] != "evdev":
            raise ValueError("Expected evdev calibration version 1.")
        if not np.isfinite(self.deadzone) or not 0 <= self.deadzone < 1:
            raise ValueError("Expected finite deadzone in [0, 1).")
        identity = cfg["device"]
        if not identity["name"] or any(
            type(identity[key]) is not int or identity[key] < 0 for key in ("vendor", "product")
        ):
            raise ValueError("Expected device name and nonnegative vendor/product IDs.")
        codes = [cfg["buttons"][name] for name in BUTTONS]
        axis_codes = []
        for name in AXES:
            spec = cfg["axes"][name]
            if spec.get("kind", "axis") == "buttons":
                if name != "yaw":
                    raise ValueError("Expected button axis only for yaw.")
                codes.extend((spec["positive"], spec["negative"]))
                continue
            if spec.get("kind", "axis") != "axis":
                raise ValueError(f"Expected axis or buttons mapping, found {spec}.")
            axis_codes.append(spec["code"])
            low, high, rest, end = (float(spec[k]) for k in ("min", "max", "rest", "positive"))
            if not np.isfinite([low, high, rest, end]).all() or not (
                low < high
                and low <= rest <= high
                and low <= end <= high
                and abs(end - rest) >= 0.15 * (high - low)
            ):
                raise ValueError(f"Expected usable finite calibration range for {name}.")
        for kind, values in (("button", codes), ("axis", axis_codes)):
            if any(type(code) is not int or code < 0 for code in values) or len(set(values)) != len(values):
                raise ValueError(f"Expected distinct nonnegative {kind} codes.")

    def decode(
        self, raw_axes: Mapping[int, float], pressed: set[int]
    ) -> tuple[dict[str, float], dict[str, bool]]:
        """Convert kernel axis values and currently pressed key codes."""
        axes = {}
        for name in AXES:
            spec = self.calibration["axes"][name]
            if spec.get("kind") == "buttons":
                value = float(spec["positive"] in pressed) - float(spec["negative"] in pressed)
            else:
                code = spec["code"]
                value = raw_axes.get(code)
                if value is None or not np.isfinite(value) or not spec["min"] <= value <= spec["max"]:
                    raise ValueError(f"Expected finite in-range input for {name}, found {value}.")
                value = (value - spec["rest"]) / (spec["positive"] - spec["rest"])
            value = float(np.clip(value, 0 if name in ("lt", "rt") else -1, 1))
            axes[name] = float(np.sign(value) * max(0, abs(value) - self.deadzone) / (1 - self.deadzone))
        return axes, {name: code in pressed for name, code in self.calibration["buttons"].items()}


class XBotInput:
    """Read a calibrated gamepad and produce normalized Cartesian increments.

    Pose rotation uses extrinsic XYZ Euler increments, scaled by the receiving
    environment. TCP-frame conversion composes rotations before recovering Euler
    components. Angular-velocity consumers explicitly request vector semantics.

    Args:
        device: Stable evdev device path, preferably under /dev/input/by-id.
        calibration: Native JSON calibration path or decoded mapping.
        precision_scale: LB speed multiplier.
        deadzone: Normalized input deadzone.
        view_hold_s: View hold duration required to end a session.
        mapping_profile: standard for full rotation; yaw_only maps right-stick
            vertical input to yaw, reverses calibrated XY translation, and
            disables roll/pitch in the selected frame.
    """

    def __init__(
        self,
        device: str,
        calibration: str | Mapping[str, Any],
        *,
        precision_scale: float = 0.5,
        deadzone: float = 0.08,
        view_hold_s: float = 0.5,
        mapping_profile: str = "standard",
    ) -> None:
        if mapping_profile not in ("standard", "yaw_only"):
            raise ValueError(f"Expected mapping_profile standard or yaw_only, found {mapping_profile!r}.")
        self._device = None
        self.mapping_profile = mapping_profile
        self.path = device
        if isinstance(calibration, str):
            calibration = json.loads(Path(calibration).expanduser().read_text())
        self.mapping = XBotMapping(calibration, deadzone)
        self._calibration_json = json.dumps(self.mapping.calibration, sort_keys=True)
        values = (
            precision_scale,
            view_hold_s,
        )
        if not np.isfinite(values).all() or any(value <= 0 for value in values) or precision_scale > 1:
            raise ValueError("Expected finite precision_scale in (0, 1] and positive finite view_hold_s.")
        self.precision_scale = precision_scale
        self.view_hold_s = view_hold_s
        self._resync_count = 0
        self._clear_state()

    @property
    def is_connected(self):
        return self._device is not None

    def connect(self):
        self._device = self._open()

    def disconnect(self):
        if self._device is not None:
            self._device.close()
            self._device = None

    def _open(self) -> Any:
        from evdev import InputDevice, ecodes

        device = InputDevice(self.path)
        try:
            expected = self.mapping.calibration["device"]
            identity = {
                "name": device.name,
                "vendor": device.info.vendor,
                "product": device.info.product,
            }
            if identity != expected:
                raise ValueError(f"Expected calibrated identity {expected}, found {identity}.")
            capabilities = device.capabilities(absinfo=False)
            keys = set(capabilities.get(ecodes.EV_KEY, []))
            required = set(self.mapping.calibration["buttons"].values())
            for spec in self.mapping.calibration["axes"].values():
                if spec.get("kind") == "buttons":
                    required.update((spec["positive"], spec["negative"]))
                else:
                    code = spec["code"]
                    if code not in capabilities.get(ecodes.EV_ABS, []):
                        raise ValueError(f"Expected calibrated axis code {code} in input device.")
                    info = device.absinfo(code)
                    if (info.min, info.max) != (spec["min"], spec["max"]):
                        raise ValueError(f"Expected calibrated range for axis {code}, found {info}.")
            if not required <= keys:
                raise ValueError(f"Expected calibrated button codes {required}, found {keys}.")
        except BaseException:
            device.close()
            raise
        self._clear_state()
        return XBotReader(device, self.mapping)

    @property
    def observation_features(self) -> Features:
        """Describe the latest decoded state and acquisition time."""
        return {
            "axes": {},
            "buttons": {},
            "received_at": {},
            "syncing": {},
            "resync_count": {},
        }

    def get_observation(self) -> Observation:
        """Return the background reader's latest snapshot."""
        if not self.is_connected:
            raise RuntimeError("Expected connected XBot before reading input.")
        return self._device.get_observation()

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Reset session state without draining or replaying device events."""
        self._clear_state()
        if self.is_connected:
            self._previous = self.get_observation()["buttons"].copy()

    def _clear_state(self) -> None:
        """Disarm without consuming input or hiding a device failure."""
        self._grip = None
        self._previous = dict.fromkeys(BUTTONS, True)
        self._released = False
        self._view_since = None
        self._view_fired = False
        self._finished = False
        self._frame = "base"

    def action(self, reading: Mapping[str, Any], context: Mapping[str, Any]) -> TeleopAction:
        """Generate normalized input and one-shot events from valid readings."""
        now = time.monotonic()
        try:
            stamp = float(reading["received_at"])
            axes = {name: float(reading["axes"][name]) for name in AXES}
            if not np.isfinite(list(axes.values())).all() or any(abs(value) > 1 for value in axes.values()):
                raise ValueError("Expected finite normalized XBot axes in [-1, 1].")
            buttons = {name: bool(reading["buttons"][name]) for name in BUTTONS}
            pose = np.asarray(context["tcp_pose"], dtype=np.float64).reshape(-1)
            if (
                pose.shape != (7,)
                or not np.isfinite(pose).all()
                or not 0.5 <= np.linalg.norm(pose[3:]) <= 1.5
            ):
                raise ValueError("Expected finite TCP xyz+xyzw with a valid quaternion.")
            rotation = Rotation.from_quat(pose[3:])
        except (KeyError, TypeError, ValueError, TimeoutError):
            self._clear_state()
            raise
        resync_count = reading.get("resync_count", self._resync_count)
        if reading.get("syncing", False) or resync_count != self._resync_count:
            self._resync_count = resync_count
            self._clear_state()
            buttons = dict.fromkeys(BUTTONS, True)
        if self._grip is None:
            self._grip = 1.0 if bool(np.asarray(context["gripper_open"]).item()) else -1.0
        edges = [name for name in BUTTONS if buttons[name] and not self._previous[name]]
        if not buttons["rb"]:
            self._released = True
        if not buttons["view"]:
            self._view_since, self._view_fired = None, False
        elif "view" in edges:
            self._view_since = now
        self._previous = buttons.copy()
        enabled = buttons["rb"] and self._released and not self._finished
        event_names = {"menu": "start", "y": "save", "b": "discard"}
        events = [event_names[name] for name in edges if name in event_names and not self._finished]
        if (
            self._view_since is not None
            and not self._view_fired
            and not self._finished
            and now - self._view_since >= self.view_hold_s
        ):
            events.append("finalize")
            self._view_fired = self._finished = True
            enabled = False
        delta = np.zeros(6)
        if enabled and edges.count("a") % 2:
            self._grip *= -1
        fixed_frame = context.get("teleop_frame")
        if fixed_frame is not None:
            self._frame = fixed_frame
        if "x" in edges and not self._finished and fixed_frame is None:
            if edges.count("x") % 2:
                self._frame = "tcp" if self._frame == "base" else "base"
        elif enabled:
            lx, ly = axes["lx"], axes["ly"]
            lx, ly = (lx, 0) if abs(lx) >= abs(ly) else (0, ly)
            v = np.array([ly, lx, axes["rt"] - axes["lt"]])
            if self.mapping_profile == "yaw_only":
                v[:2] *= -1
                w = np.array([0.0, 0.0, np.clip(axes["ry"] + axes["yaw"], -1, 1)])
            else:
                rx, ry = axes["rx"], axes["ry"]
                rx, ry = (rx, 0) if abs(rx) >= abs(ry) else (0, ry)
                w = np.array([ry, rx, axes["yaw"]])
            scale = self.precision_scale if buttons["lb"] else 1.0
            dp = v * scale
            dr = w * scale
            if self._frame == "tcp":
                dp = rotation.apply(dp)
                if context.get("rotation_action_type", "euler_xyz") == "angular_velocity":
                    dr = rotation.apply(dr)
                else:
                    angular_scale = float(context["action_scale"][1])
                    if not np.isfinite(angular_scale) or angular_scale <= 0:
                        raise ValueError("Expected positive finite Euler action scale.")
                    # Change the incremental rotation frame, not its Euler components.
                    increment = Rotation.from_euler("xyz", dr * angular_scale)
                    dr = (rotation * increment * rotation.inv()).as_euler("xyz") / angular_scale
            delta = np.r_[dp, dr]
        return TeleopAction(
            parts={
                "arm": delta.astype(np.float32),
                "end_effector": np.array([self._grip], dtype=np.float32),
            },
            driving=bool(enabled),
            info={
                "xbot_events": events,
                "xbot_delta": delta.tolist(),
                "xbot_frame": self._frame,
                "xbot_received_at": stamp,
                "xbot_input_age": now - stamp,
                "xbot_enabled": bool(enabled),
                "xbot_calibration_json": self._calibration_json,
                "xbot_max_age": None,
            },
        )


class XBotReader:
    """Own an evdev handle and publish only its latest decoded state."""

    def __init__(self, device: Any, mapping: XBotMapping) -> None:
        self._device = device
        self._mapping = mapping
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._syncing = False
        self._resync_count = 0
        try:
            self._latest = self._snapshot()
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
        except BaseException:
            device.close()
            raise

    def _snapshot(self) -> Observation:
        raw = {
            spec["code"]: self._device.absinfo(spec["code"]).value
            for spec in self._mapping.calibration["axes"].values()
            if spec.get("kind") != "buttons"
        }
        axes, buttons = self._mapping.decode(raw, set(self._device.active_keys()))
        return {"axes": axes, "buttons": buttons, "received_at": time.monotonic()}

    def _read(self) -> None:
        from evdev import ecodes

        syncing = False
        try:
            while not self._stop.is_set():
                try:
                    for event in self._device.read():
                        if event.type == ecodes.EV_SYN:
                            if event.code == ecodes.SYN_DROPPED:
                                syncing = True
                                with self._lock:
                                    self._syncing = True
                                    self._resync_count += 1
                            elif event.code == ecodes.SYN_REPORT:
                                syncing = False
                except BlockingIOError:
                    pass
                if not syncing:
                    snapshot = self._snapshot()
                    with self._lock:
                        self._latest = snapshot
                        self._syncing = False
                # Poll independently of collection, without spinning on idle devices.
                self._stop.wait(0.002)
        except Exception as error:
            with self._lock:
                self._error = error

    def get_observation(self) -> Observation:
        """Copy current state, propagating background device failures."""
        with self._lock:
            if self._error is not None:
                raise self._error
            return {
                **copy.deepcopy(self._latest),
                "syncing": self._syncing,
                "resync_count": self._resync_count,
            }

    def close(self) -> None:
        """Stop the reader before releasing its device handle."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._device is not None:
            device, self._device = self._device, None
            device.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError("XBot reader did not stop after closing the device.")
