# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared operator loop for the two UR examples."""

import json
import logging
import time
from dataclasses import asdict

from lerobot.robots.ur import UR
from lerobot.teleoperators.xbot import XBot, to_ur_action

logger = logging.getLogger(__name__)


def open_dataset(cfg, robot):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import hw_to_dataset_features

    features = hw_to_dataset_features(robot.observation_features, "observation", use_video=cfg.USE_VIDEOS)
    features.update(hw_to_dataset_features(robot.action_features, "action", use_video=cfg.USE_VIDEOS))
    manifest = {"robot": asdict(cfg.ROBOT), "task": asdict(cfg.TASK), "task_type": type(cfg.TASK).__name__}
    manifest = json.loads(json.dumps(manifest, default=str))
    manifest_path = cfg.DATASET_ROOT / "ur_setup.json"
    if cfg.RESUME:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume requires matching robot/task settings in ur_setup.json")
        dataset = LeRobotDataset.resume(repo_id=cfg.REPO_ID, root=cfg.DATASET_ROOT)
        if dataset.fps != cfg.FPS or any(
            any(
                dataset.features.get(k, {}).get(field) != v.get(field)
                for field in ("dtype", "shape", "names")
            )
            for k, v in features.items()
        ):
            dataset.finalize()
            raise ValueError("Resume requires the same FPS, mode, cameras, gripper and observation features")
        return dataset
    dataset = LeRobotDataset.create(
        repo_id=cfg.REPO_ID,
        root=cfg.DATASET_ROOT,
        fps=cfg.FPS,
        features=features,
        robot_type=robot.name,
        use_videos=cfg.USE_VIDEOS,
        image_writer_threads=4 if robot.cameras else 0,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return dataset


def run(cfg, *, record=False):
    """Run Menu/RB/Y/B/View controls; save only explicitly successful episodes."""
    if not cfg.ENABLE_MOTION:
        raise ValueError("Review config.py and set ENABLE_MOTION=True to enable robot motion")
    if cfg.FPS <= 0 or cfg.NUM_EPISODES <= 0:
        raise ValueError("Expected positive FPS and NUM_EPISODES")
    robot, teleop = UR(cfg.ROBOT), XBot(cfg.TELEOP)
    from visualization import URRerunDisplay

    display = URRerunDisplay(
        enabled=cfg.DISPLAY_DATA, fps=cfg.DISPLAY_FPS, ip=cfg.DISPLAY_IP, port=cfg.DISPLAY_PORT
    )
    dataset = None
    phase, last_status, session_error = "WAITING", None, None

    def show(observation=None, *, force=False):
        nonlocal last_status
        rb = bool(phase == "RUNNING" and active and teleop.driving)
        diagnostics = robot.get_diagnostics()
        if session_error:
            diagnostics["error"] = session_error
        status = (phase, rb, diagnostics["robot_mode"], diagnostics["safety_mode"])
        if status != last_status:
            logger.info("UR state=%s RB=%s robot=%s safety=%s", *status)
            last_status = status
        display.update(diagnostics, state=phase, rb=rb, observation=observation, force=force)

    active, needs_reset, frames, saved = False, True, 0, 0
    try:
        if record:
            dataset = open_dataset(cfg, robot)
        robot.connect()
        teleop.connect()
        logger.info("Menu: start/pause; RB: enable; Y: save; B: discard; hold View: exit")
        while True:
            started = time.monotonic()
            observation = robot.get_observation()
            feedback = observation | {
                "rotation_scale": cfg.ROTATION,
                "velocity_mode": cfg.ROBOT.mode.endswith("velocity"),
                "joint_mode": cfg.ROBOT.mode.startswith("joint"),
                "gripper.open": observation.get("gripper.width", 1.0)
                >= (cfg.ROBOT.gripper.max_width / 2 if cfg.ROBOT.gripper else 0.5),
            }
            teleop.send_feedback(feedback)
            action = teleop.get_action()
            events = set(teleop.events)
            if "finalize" in events:
                break
            if events & {"save", "discard"}:
                robot.stop()
                phase = "SAVING" if "save" in events and "discard" not in events else "DISCARDING"
                show(observation, force=True)
                if dataset is not None and frames:
                    if "discard" in events:
                        dataset.clear_episode_buffer()
                    else:
                        dataset.save_episode(parallel_encoding=False)
                        saved += 1
                        logger.info("Saved episode %d (%d frames)", saved, frames)
                frames, active, needs_reset = 0, False, True
                teleop.reset()
                if record and saved >= cfg.NUM_EPISODES:
                    break
            elif "start" in events:
                robot.stop()
                if not active and needs_reset:
                    phase = "RESETTING"
                    show(observation, force=True)
                    cfg.TASK.reset(robot)
                    needs_reset = False
                active = not active
                teleop.reset()  # Require a fresh RB release after pause/reset.
            elif active and teleop.driving:
                command = to_ur_action(
                    robot,
                    action,
                    observation,
                    translation=cfg.TRANSLATION,
                    rotation=cfg.ROTATION,
                    joint_axis=teleop.joint_axis,
                    joint_index=cfg.JOINT_INDEX,
                    joint_step=cfg.JOINT_STEP,
                )
                accepted = robot.send_action(command)
                if dataset is not None:
                    from lerobot.utils.feature_utils import build_dataset_frame

                    frame = build_dataset_frame(dataset.features, observation, "observation")
                    frame.update(build_dataset_frame(dataset.features, accepted, "action"))
                    frame["task"] = cfg.TASK.description
                    dataset.add_frame(frame)
                    frames += 1
            else:
                robot.stop()
            phase = (
                "RUNNING"
                if active and teleop.driving
                else ("PAUSED" if not active and not needs_reset else "WAITING")
            )
            show(observation)
            time.sleep(max(0.0, 1 / cfg.FPS - (time.monotonic() - started)))
    except BaseException as error:
        session_error = str(error)
        phase = "STOPPED" if isinstance(error, KeyboardInterrupt) else "FAULT"
        logger.error("UR session ended: %s", error)
        # No viewer operation may delay the stop request on an exception.
        try:
            robot.stop()
        finally:
            try:
                show(force=True)
            except Exception:
                logger.exception("Could not publish final UR status")
        raise
    finally:
        # Attempt all cleanup, retaining the first failure (robot stop has priority).
        operations = [("robot disconnect", robot.disconnect), ("teleoperator disconnect", teleop.disconnect)]
        if dataset is not None:
            if frames:
                operations.append(("discard unsaved episode", dataset.clear_episode_buffer))
            operations.append(("finalize dataset", dataset.finalize))
        cleanup_error = None
        for label, operation in operations:
            try:
                operation()
            except BaseException as error:
                logger.exception("UR cleanup failed during %s", label)
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None:
            phase, session_error = "FAULT", str(cleanup_error)
        elif phase != "FAULT":
            phase = "STOPPED"
        try:
            show(force=True)
        except Exception:
            logger.exception("Could not publish final UR status")
        try:
            display.close()
        except Exception:
            logger.exception("Could not close Rerun")
        if cleanup_error is not None:
            raise cleanup_error
