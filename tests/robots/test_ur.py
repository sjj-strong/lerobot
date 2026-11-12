# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N802  # SDK fakes preserve the vendor method names.
import numpy as np
import pytest

pytest.importorskip("scipy")

from lerobot.robots.ur import UR, URConfig
from lerobot.robots.ur.config_ur import ForceConfig
from lerobot.robots.ur.controller import force_wrench, limit_target

pytestmark = pytest.mark.hardware


def test_features_without_optional_hardware():
    robot = UR(URConfig(robot_ip="127.0.0.1"))
    assert len(robot.observation_features) == 18
    assert len(robot.action_features) == 6
    assert not robot.is_connected


def test_mode_validation():
    with pytest.raises(ValueError):
        URConfig(robot_ip="127.0.0.1", mode="joint_impedance")


def test_target_limits_and_invalid_input():
    cfg = URConfig(robot_ip="127.0.0.1", mode="joint_position")
    np.testing.assert_allclose(limit_target(cfg, np.ones(6), np.zeros(6)), 0.008)
    with pytest.raises(ValueError):
        limit_target(cfg, np.full(6, np.nan), np.zeros(6))


def test_tcp_feedforward_rotates_then_clips():
    cfg = ForceConfig(
        stiffness=(0,) * 6, damping=(0,) * 6, tcp_wrench=(0, 0, -80, 0, 0, 0), max_wrench=(40,) * 6
    )
    pose = np.array([0, 0, 0, 0, np.pi / 2, 0])
    wrench = force_wrench(cfg, pose, pose, np.zeros(6))
    np.testing.assert_allclose(wrench, [-40, 0, 0, 0, 0, 0], atol=1e-10)


def test_force_pd_and_selection():
    cfg = ForceConfig(stiffness=(10,) * 6, damping=(2,) * 6, selection_vector=(1, 1, 1, 0, 0, 0))
    result = force_wrench(cfg, np.array([0.1, 0, 0, 0, 0, 0]), np.zeros(6), np.array([0.1, 0, 0, 0, 0, 1]))
    np.testing.assert_allclose(result, [0.8, 0, 0, 0, 0, 0])


def test_all_modes_dispatch():
    from lerobot.robots.ur.controller import RTDEController

    class SDK:
        def __getattr__(self, name):
            return lambda *args: (name, args)

    expected = {
        "joint_position": "servoJ",
        "joint_velocity": "speedJ",
        "cartesian_position": "servoL",
        "cartesian_velocity": "speedL",
        "cartesian_force": "forceMode",
    }
    for mode, method in expected.items():
        controller = RTDEController(URConfig(robot_ip="test", mode=mode))
        controller.control = SDK()
        result = controller._write(np.zeros(6), {"tcp": np.zeros(6), "velocity": np.zeros(6)})
        assert result[0] == method


def test_xbot_processor_physical_action():
    from lerobot.teleoperators.xbot import to_ur_action

    robot = UR(URConfig(robot_ip="test", mode="cartesian_position"))
    observation = dict.fromkeys(robot.observation_features, 0.0)
    result = to_ur_action(
        robot,
        {"dx": 1.0, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0},
        observation,
        translation=0.005,
        rotation=0.02,
    )
    assert result["tcp.x"] == 0.005


class FakeReceive:
    def isConnected(self):
        return True

    def isProtectiveStopped(self):
        return False

    def isEmergencyStopped(self):
        return False

    def getActualQ(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return [0.0] * 6

    def getActualTCPSpeed(self):
        return [0.0] * 6

    def getActualTCPForce(self):
        return [0.0] * 6

    def getTimestamp(self):
        return 1.0

    def getSafetyMode(self):
        return 1

    def getRobotMode(self):
        return 7


def test_stalled_feedback_rejected():
    import time

    from lerobot.robots.ur.controller import RTDEController

    controller = RTDEController(URConfig(robot_ip="test", mode="cartesian_position", command_timeout=0.01))
    controller.receive = FakeReceive()
    controller.state()
    time.sleep(0.02)
    with pytest.raises(TimeoutError, match="feedback"):
        controller.state()


def test_factory_and_optional_features():
    from lerobot.robots.ur import RobotiqConfig
    from lerobot.robots.utils import make_robot_from_config

    robot = make_robot_from_config(
        URConfig(robot_ip="test", include_wrench=False, gripper=RobotiqConfig(port="/dev/test"))
    )
    assert len(robot.observation_features) == 13
    assert len(robot.action_features) == 7


def test_command_watchdog_stops_sdk():
    import time

    from lerobot.robots.ur.controller import RTDEController

    class SDK:
        stopped = False

        def initPeriod(self):
            return None

        def waitPeriod(self, tick):
            time.sleep(0.002)

        def servoL(self, *args):
            return True

        def servoStop(self, *args):
            self.stopped = True
            return True

    class Receiver(FakeReceive):
        def getTimestamp(self):
            return time.monotonic()

    controller = RTDEController(URConfig(robot_ip="test", mode="cartesian_position", command_timeout=0.01))
    controller.receive, controller.control = Receiver(), SDK()
    controller.send(np.zeros(6))
    deadline = time.monotonic() + 1
    while controller._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert controller.control.stopped
    with pytest.raises(RuntimeError, match="control loop failed"):
        controller.send(np.zeros(6))
    controller.stop()


@pytest.mark.dataset
def test_native_dataset_roundtrip(tmp_path):
    pytest.importorskip("datasets")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    robot = UR(URConfig(robot_ip="test", mode="cartesian_position"))
    features = hw_to_dataset_features(robot.observation_features, "observation")
    features.update(hw_to_dataset_features(robot.action_features, "action"))
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/ur", root=root, fps=30, features=features, robot_type="ur", use_videos=False
    )
    observation = dict.fromkeys(robot.observation_features, 0.0)
    action = dict.fromkeys(robot.action_features, 0.001)
    for _ in range(2):
        dataset.add_frame(
            build_dataset_frame(dataset.features, observation, "observation")
            | build_dataset_frame(dataset.features, action, "action")
            | {"task": "test UR"}
        )
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    loaded = LeRobotDataset("local/ur", root=root, video_backend="pyav")
    np.testing.assert_allclose(loaded[0]["action"], 0.001)
    assert loaded.num_episodes == 1
    assert loaded.features["observation.state"]["names"][:6] == list(robot.observation_features)[:6]


@pytest.mark.dataset
@pytest.mark.parametrize("camera_count", [0, 1, 3])
@pytest.mark.parametrize("use_videos", [False, True])
def test_configured_images_and_resume(tmp_path, monkeypatch, camera_count, use_videos):
    import importlib.util
    from dataclasses import dataclass
    from pathlib import Path
    from types import SimpleNamespace

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    pytest.importorskip("datasets")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame

    source = Path(__file__).resolve().parents[2] / "examples/ur/session.py"
    spec = importlib.util.spec_from_file_location("ur_session_example", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @dataclass
    class Task:
        description: str = "test"

    cfg = SimpleNamespace(
        ROBOT=URConfig(
            robot_ip="test",
            cameras={
                f"cam{i}": OpenCVCameraConfig(index_or_path=i, width=32, height=32, fps=30)
                for i in range(camera_count)
            },
        ),
        TASK=Task(),
        USE_VIDEOS=use_videos,
        DATASET_ROOT=tmp_path / "dataset",
        RESUME=False,
        FPS=30,
        REPO_ID="local/ur",
    )
    robot = UR(cfg.ROBOT)
    dataset = module.open_dataset(cfg, robot)
    observation = {
        key: np.zeros(value, dtype=np.uint8) if isinstance(value, tuple) else 0.0
        for key, value in robot.observation_features.items()
    }
    action = dict.fromkeys(robot.action_features, 0.001)
    frame = (
        build_dataset_frame(dataset.features, observation, "observation")
        | build_dataset_frame(dataset.features, action, "action")
        | {"task": "test"}
    )
    dataset.add_frame(frame.copy())
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    cfg.RESUME = True
    dataset = module.open_dataset(cfg, robot)
    dataset.add_frame(frame.copy())
    dataset.clear_episode_buffer()
    dataset.add_frame(frame.copy())
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    loaded = LeRobotDataset("local/ur", root=cfg.DATASET_ROOT, video_backend="pyav")
    assert loaded.num_episodes == 2
    for i in range(camera_count):
        assert loaded[0][f"observation.images.cam{i}"].shape == (3, 32, 32)


def test_force_target_latches_released_axes():
    from lerobot.teleoperators.xbot import to_ur_action

    robot = UR(URConfig(robot_ip="test", mode="cartesian_force"))
    obs = dict.fromkeys(robot.observation_features, 0.0)
    neutral = dict.fromkeys(("dx", "dy", "dz", "drx", "dry", "drz"), 0.0)
    to_ur_action(robot, neutral | {"dx": 1.0}, obs)
    obs["tcp.x"] = 0.003
    released = to_ur_action(robot, neutral, obs)
    assert released["force_target.x"] == pytest.approx(0.003)
    obs["tcp.x"] = 0.008
    held = to_ur_action(robot, neutral | {"dy": 1.0}, obs)
    assert held["force_target.x"] == pytest.approx(0.003)
    robot.stop()
    restarted = to_ur_action(robot, neutral, obs)
    assert restarted["force_target.x"] == pytest.approx(0.008)


def test_failed_force_stop_still_attempts_linear_stop():
    from lerobot.robots.ur.controller import RTDEController

    class SDK:
        stopped = False

        def forceModeStop(self):
            return False

        def stopL(self, *args):
            self.stopped = True
            return True

    controller = RTDEController(URConfig(robot_ip="test", mode="cartesian_force"))
    controller.control = SDK()
    with pytest.raises(RuntimeError):
        controller._stop_sdk()
    assert controller.control.stopped


def test_no_expired_write_after_slow_feedback():
    import time

    from lerobot.robots.ur.controller import RTDEController

    class Receiver(FakeReceive):
        reads = 0

        def getActualQ(self):
            self.reads += 1
            if self.reads > 1:
                time.sleep(0.03)
            return [0.0] * 6

        def getTimestamp(self):
            return time.monotonic()

    class SDK:
        writes = 0

        def initPeriod(self):
            return None

        def waitPeriod(self, tick):
            pass

        def servoL(self, *args):
            self.writes += 1
            return True

        def servoStop(self, *args):
            return True

    controller = RTDEController(URConfig(robot_ip="test", mode="cartesian_position", command_timeout=0.01))
    controller.receive, controller.control = Receiver(), SDK()
    controller.send(np.zeros(6))
    deadline = time.monotonic() + 1
    while controller._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert controller.control.writes == 0
    controller.stop()


def test_reconnect_clears_fault_and_feedback_epoch(monkeypatch):
    import sys
    from types import SimpleNamespace

    from lerobot.robots.ur.controller import RTDEController

    class Receiver(FakeReceive):
        def disconnect(self):
            pass

    class SDK:
        def disconnect(self):
            pass

    monkeypatch.setitem(sys.modules, "rtde_control", SimpleNamespace(RTDEControlInterface=lambda *a: SDK()))
    monkeypatch.setitem(
        sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=lambda *a: Receiver())
    )
    controller = RTDEController(URConfig(robot_ip="test", mode="cartesian_position"))
    controller.connect()
    controller._error = TimeoutError("old failure")
    controller._feedback_stamp = 100
    controller.disconnect()
    controller.connect()
    assert controller.state()["tcp"].shape == (6,)
    controller.disconnect()


def test_default_force_mode_and_critical_damping():
    cfg = URConfig(robot_ip="test")
    assert cfg.mode == "cartesian_force"
    assert cfg.force.damping is None
    np.testing.assert_allclose(cfg.force.resolved_damping, 2 * np.sqrt(cfg.force.stiffness))
    cfg.force.stiffness = (0, 1, 4, 9, 16, 25)
    np.testing.assert_allclose(cfg.force.resolved_damping, [0, 2, 4, 6, 8, 10])
    force = force_wrench(cfg.force, np.zeros(6), np.zeros(6), np.full(6, 0.01))
    np.testing.assert_allclose(force, -np.array([0, 2, 4, 6, 8, 10]) * 0.01 * cfg.force.selection_vector)


def test_explicit_damping_remains_supported():
    cfg = ForceConfig(damping=(1, 2, 3, 4, 5, 6))
    np.testing.assert_array_equal(cfg.resolved_damping, [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        ForceConfig(stiffness=(-1, 0, 0, 0, 0, 0))


def test_diagnostics_do_not_change_dataset_features():
    robot = UR(URConfig(robot_ip="test", mode="cartesian_position"))
    features = robot.observation_features.copy()
    diagnostics = robot.get_diagnostics()
    assert diagnostics["target_count"] == 0
    assert diagnostics["control_count"] == 0
    assert diagnostics["robot_mode"] is None
    assert diagnostics["safety_mode"] is None
    assert robot.observation_features == features


def test_control_diagnostics_count_successful_sdk_writes():
    import time

    from lerobot.robots.ur.controller import RTDEController

    class SDK:
        writes = 0

        def initPeriod(self):
            return None

        def waitPeriod(self, tick):
            time.sleep(0.002)

        def servoL(self, *args):
            self.writes += 1
            return True

        def servoStop(self, *args):
            return True

    class Receiver(FakeReceive):
        def getTimestamp(self):
            return time.monotonic()

    controller = RTDEController(URConfig(robot_ip="test", mode="cartesian_position"))
    controller.receive, controller.control = Receiver(), SDK()
    controller.send(np.zeros(6))
    time.sleep(0.02)
    controller.stop()
    data = controller.get_diagnostics()
    assert data["target_count"] == 1
    assert data["control_count"] == controller.control.writes > 0
    assert data["robot_mode"] == 7
    assert data["safety_mode"] == 1
    np.testing.assert_array_equal(data["wrench"], np.zeros(6))
    count = data["control_count"]
    time.sleep(0.01)
    assert controller.get_diagnostics()["control_count"] == count


@pytest.mark.viz
def test_rerun_diagnostics_rates_and_six_axis_wrench(monkeypatch):
    import importlib.util
    from pathlib import Path

    rr = pytest.importorskip("rerun")
    path = Path(__file__).resolve().parents[2] / "examples/ur/visualization.py"
    spec = importlib.util.spec_from_file_location("ur_visualization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(rr, "spawn", lambda **kwargs: None)
    logged = {}
    blueprints = []
    send_blueprint = rr.send_blueprint

    def capture_blueprint(blueprint, **kwargs):
        blueprints.append(blueprint)
        return send_blueprint(blueprint, **kwargs)

    monkeypatch.setattr(rr, "send_blueprint", capture_blueprint)
    original = rr.log

    def capture(path, entity=None, *args, **kwargs):
        if entity is not None and hasattr(entity, "scalars"):
            logged[path] = entity.scalars.as_arrow_array().to_pylist()
        return original(path, entity, *args, **kwargs)

    monkeypatch.setattr(rr, "log", capture)
    display = module.URRerunDisplay(enabled=True, fps=10)
    memory = rr.memory_recording()
    sample = {
        "sampled_at": 10.0,
        "target_count": 0,
        "control_count": 0,
        "wrench": np.arange(6, dtype=float),
        "robot_mode": 7,
        "safety_mode": 1,
        "feedback_age_s": 0.0,
        "error": None,
    }
    try:
        display.update(sample, state="WAITING", rb=False)
        initial_layouts = len(blueprints)
        display.update(
            sample | {"sampled_at": 11.0, "target_count": 30, "control_count": 500}, state="RUNNING", rb=True
        )
        assert len(blueprints) == initial_layouts
        assert logged["observation.rates/target_hz"] == [30.0]
        assert logged["observation.rates/control_hz"] == [500.0]
        for i, name in enumerate(("Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm")):
            assert logged["observation.wrench/" + name] == [float(i)]
        display.update(
            sample | {"sampled_at": 12.0, "target_count": 30, "control_count": 500}, state="PAUSED", rb=False
        )
        assert logged["observation.rates/target_hz"] == [0.0]
        assert logged["observation.rates/control_hz"] == [0.0]
        assert memory.num_msgs() > 0
    finally:
        display.close()


@pytest.mark.viz
def test_viewer_failure_does_not_mask_connection_failure(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from lerobot.teleoperators.xbot import XBotConfig

    rr = pytest.importorskip("rerun")
    example_dir = Path(__file__).resolve().parents[2] / "examples/ur"
    monkeypatch.syspath_prepend(str(example_dir))
    spec = importlib.util.spec_from_file_location("ur_session_fault", example_dir / "session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calibration = {
        "version": 1,
        "backend": "evdev",
        "device": {"name": "test", "vendor": 1, "product": 2},
        "axes": {
            name: {"code": i, "rest": 0, "positive": 100, "min": -100, "max": 100}
            for i, name in enumerate(("lx", "ly", "rx", "ry", "lt", "rt", "yaw"))
        },
        "buttons": {name: 300 + i for i, name in enumerate(("rb", "lb", "a", "x", "menu", "y", "b", "view"))},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))

    class Receive(FakeReceive):
        def disconnect(self):
            pass

    def unavailable_robot(*args):
        raise ConnectionError("robot unavailable")

    def broken_viewer(*args, **kwargs):
        raise RuntimeError("viewer failed")

    monkeypatch.setitem(sys.modules, "rtde_control", SimpleNamespace(RTDEControlInterface=unavailable_robot))
    monkeypatch.setitem(
        sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=lambda *args: Receive())
    )
    monkeypatch.setattr(rr, "spawn", lambda **kwargs: None)
    monkeypatch.setattr(rr, "log", broken_viewer)
    shutdown = rr.rerun_shutdown
    monkeypatch.setattr(rr, "rerun_shutdown", broken_viewer)
    cfg = SimpleNamespace(
        ENABLE_MOTION=True,
        FPS=30,
        NUM_EPISODES=1,
        ROBOT=URConfig(robot_ip="test"),
        TELEOP=XBotConfig(device="unused", calibration_path=str(path)),
        DISPLAY_DATA=True,
        DISPLAY_FPS=10,
        DISPLAY_IP=None,
        DISPLAY_PORT=None,
    )
    try:
        with pytest.raises(ConnectionError, match="robot unavailable"):
            module.run(cfg)
    finally:
        shutdown()


@pytest.mark.viz
def test_normal_exit_with_disconnect_failure_displays_fault(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from lerobot.teleoperators.xbot import XBotConfig

    rr = pytest.importorskip("rerun")
    evdev = pytest.importorskip("evdev")
    example_dir = Path(__file__).resolve().parents[2] / "examples/ur"
    monkeypatch.syspath_prepend(str(example_dir))
    spec = importlib.util.spec_from_file_location("ur_session_cleanup", example_dir / "session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calibration = {
        "version": 1,
        "backend": "evdev",
        "device": {"name": "test", "vendor": 1, "product": 2},
        "axes": {
            name: {"code": i, "rest": 0, "positive": 100, "min": -100, "max": 100}
            for i, name in enumerate(("lx", "ly", "rx", "ry", "lt", "rt", "yaw"))
        },
        "buttons": {name: 300 + i for i, name in enumerate(("rb", "lb", "a", "x", "menu", "y", "b", "view"))},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))

    class Receive(FakeReceive):
        def getTimestamp(self):
            return time.monotonic()

        def disconnect(self):
            pass

    class Control:
        def disconnect(self):
            raise RuntimeError("disconnect failed")

    class Input:
        name = "test"
        info = SimpleNamespace(vendor=1, product=2)

        def __init__(self, *args):
            self.started = time.monotonic()

        def capabilities(self, **kwargs):
            return {evdev.ecodes.EV_ABS: list(range(7)), evdev.ecodes.EV_KEY: list(range(300, 308))}

        def absinfo(self, code):
            return SimpleNamespace(value=0, min=-100, max=100)

        def active_keys(self):
            return [307] if time.monotonic() - self.started > 0.1 else []

        def read(self):
            return []

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "rtde_control", SimpleNamespace(RTDEControlInterface=lambda *a: Control())
    )
    monkeypatch.setitem(
        sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=lambda *a: Receive())
    )
    monkeypatch.setattr(evdev, "InputDevice", Input)
    monkeypatch.setattr(rr, "spawn", lambda **kwargs: None)
    statuses = []
    original = rr.log

    def capture(path, entity=None, *args, **kwargs):
        if path == "ur/status":
            statuses.extend(entity.text.as_arrow_array().to_pylist())
        return original(path, entity, *args, **kwargs)

    monkeypatch.setattr(rr, "log", capture)
    cfg = SimpleNamespace(
        ENABLE_MOTION=True,
        FPS=30,
        NUM_EPISODES=1,
        ROBOT=URConfig(robot_ip="test", mode="cartesian_position"),
        TELEOP=XBotConfig(device="unused", calibration_path=str(path), view_hold_s=0.01),
        DISPLAY_DATA=True,
        DISPLAY_FPS=10,
        DISPLAY_IP=None,
        DISPLAY_PORT=None,
        ROTATION=0.01,
    )
    with pytest.raises(RuntimeError, match="disconnect failed"):
        module.run(cfg)
    assert statuses[-1].startswith("# FAULT")
    assert "disconnect failed" in statuses[-1]
    assert "RB enabled: **False**" in statuses[-1]
