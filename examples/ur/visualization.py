# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""UR diagnostics displayed through LeRobot's Rerun backend."""

import math

import numpy as np


class URRerunDisplay:
    """Render windowed rates and measured feedback outside the RTDE thread."""

    def __init__(self, *, enabled: bool, fps: float = 10, ip: str | None = None, port: int | None = None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Expected positive finite visualization FPS")
        if (ip is None) != (port is None):
            raise ValueError("Set both DISPLAY_IP and DISPLAY_PORT for a remote Rerun viewer")
        self.enabled = enabled
        self._period = 1 / fps
        self._last_emit = -math.inf
        self._previous = None
        self._rates = {"target_hz": 0.0, "control_hz": 0.0}
        self._status = None
        self._layout_keys = None
        if enabled:
            from lerobot.utils.visualization_utils import init_visualization

            init_visualization("rerun", session_name="ur_teleoperation", ip=ip, port=port)

    def update(
        self, diagnostics: dict, *, state: str, rb: bool, observation: dict | None = None, force: bool = False
    ) -> None:
        """Log rates over one-second windows and all six measured wrench axes."""
        if not self.enabled:
            return
        now = diagnostics["sampled_at"]
        if self._previous is None or any(
            diagnostics[k] < self._previous[k] for k in ("target_count", "control_count")
        ):
            self._previous = diagnostics.copy()
            self._rates = {"target_hz": 0.0, "control_hz": 0.0}
        elapsed = now - self._previous["sampled_at"]
        if elapsed >= 1.0:
            self._rates = {
                name + "_hz": (diagnostics[name + "_count"] - self._previous[name + "_count"]) / elapsed
                for name in ("target", "control")
            }
            self._previous = diagnostics.copy()
        status = (state, rb, diagnostics["robot_mode"], diagnostics["safety_mode"], diagnostics["error"])
        if not force and status == self._status and now - self._last_emit < self._period:
            return
        self._status, self._last_emit = status, now
        import rerun as rr
        import rerun.blueprint as rrb

        from lerobot.utils.visualization_utils import log_visualization_data

        values = {"rates/" + name: value for name, value in self._rates.items()}
        wrench = diagnostics["wrench"]
        if wrench is not None:
            values.update(
                {
                    "wrench/" + name: float(value)
                    for name, value in zip(
                        ("Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm"), wrench, strict=True
                    )
                }
            )
        images = {
            name: value
            for name, value in (observation or {}).items()
            if isinstance(value, np.ndarray) and value.ndim == 3
        }
        values.update(images)
        log_visualization_data("rerun", observation=values, compress_images=True)
        robot = {7: "RUNNING"}.get(diagnostics["robot_mode"], "UNKNOWN")
        safety = {1: "NORMAL", 2: "REDUCED"}.get(diagnostics["safety_mode"], "OTHER/UNKNOWN")
        age = diagnostics["feedback_age_s"]
        age_text = "N/A" if age is None else f"{age:.3f} s"
        rr.log(
            "ur/status",
            rr.TextDocument(
                f"# {state}\n\nRB enabled: **{rb}**\n\n"
                f"UR last feedback: robot **{robot} ({diagnostics['robot_mode']})**, "
                f"safety **{safety} ({diagnostics['safety_mode']})**\n\n"
                f"Feedback age: {age_text}\n\n"
                f"target_hz: **{self._rates['target_hz']:.1f}** · control_hz: **{self._rates['control_hz']:.1f}**\n\n"
                f"Wrench frame: **base**, force **N**, torque **Nm**\n\n"
                f"{diagnostics['error'] or ''}",
                media_type="text/markdown",
            ),
        )
        layout_keys = frozenset(values)
        if layout_keys == self._layout_keys:
            return
        self._layout_keys = layout_keys
        # Override the generic grid only when entities change; retain user layout edits.
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Grid(
                    rrb.TextDocumentView(origin="ur/status", name="Status"),
                    rrb.TimeSeriesView(name="Frequency (Hz)", contents=["observation.rates/**"]),
                    rrb.TimeSeriesView(
                        name="Measured force (N)",
                        contents=[
                            "observation.wrench/Fx_N",
                            "observation.wrench/Fy_N",
                            "observation.wrench/Fz_N",
                        ],
                    ),
                    rrb.TimeSeriesView(
                        name="Measured torque (Nm)",
                        contents=[
                            "observation.wrench/Tx_Nm",
                            "observation.wrench/Ty_Nm",
                            "observation.wrench/Tz_Nm",
                        ],
                    ),
                    *[rrb.Spatial2DView(origin="observation." + name, name=name) for name in images],
                )
            )
        )

    def close(self) -> None:
        """Flush and release the official visualization backend."""
        if self.enabled:
            from lerobot.utils.visualization_utils import shutdown_visualization

            self.enabled = False
            shutdown_visualization("rerun")
