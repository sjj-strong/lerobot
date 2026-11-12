# UR and XBot with LeRobot

These examples use native LeRobot `Robot`, `Teleoperator`, camera factories and
`LeRobotDataset`. They do not import RLinf, Ray or Pinocchio. The RTDE controller
streams at 500 Hz; the operator/record loop defaults to 30 Hz. Real robot motion
has not been validated by the automated tests.

## Environment and execution

Inside the RLinf UR container, use the independent Python 3.12 environment:

```bash
cd /workspace/RLinf/external/lerobot
uv pip install --python /opt/venv/lerobot/bin/python --torch-backend cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' -e '.[ur,xbot,robotiq,dataset]'
cd /workspace/RLinf
/opt/venv/lerobot/bin/python external/lerobot/examples/ur/teleoperate.py
/opt/venv/lerobot/bin/python external/lerobot/examples/ur/record.py
```

Run one script at a time. Edit `config.py` first: set the robot IP, stable gamepad
path, existing version-1 evdev calibration JSON, task, control mode, data location
and `ENABLE_MOTION=True`. Paths are relative to the working directory.
The container needs access to the robot network and configured input, camera and
serial devices. `/opt/venv/gr00t/bin/python` is Python 3.11 and cannot run this
LeRobot version. On a host, install this checkout into a Python >=3.12 environment
before using the same script paths. An older `.venv-ur` must not be reused blindly.

## Configuration

`URConfig` supports `joint_position`, `joint_velocity`, `cartesian_position`,
`cartesian_velocity`, and `cartesian_force`. Impedance/direct-torque control is
excluded. Joint modes use left-stick Y to control `JOINT_INDEX` (0 through 5).
`TRANSLATION`, `ROTATION` and `JOINT_STEP` are per-frame increments for position
and force modes, and SI velocities for velocity modes.

`cameras={}` records state only. Add any named LeRobot camera configuration:

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.ur import RobotiqConfig

ROBOT.cameras = {
    "front": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=30),
    "wrist": RealSenseCameraConfig(serial_number_or_name="YOUR_SERIAL", width=640, height=480, fps=30),
}
ROBOT.gripper = RobotiqConfig(port="/dev/ttyUSB0", max_width=0.085)
```

Install `.[intelrealsense]` for RealSense. `gripper=None` omits the connection and
all gripper features. A configured Robotiq activates when the robot connects;
A toggles its opening while RB is held. `payload_mass=None` preserves the UR
controller setting; otherwise set mass in kg and flange-relative CoG in metres.
No payload is inferred from historical RLinf experiments.

Force mode uses native type 2. In `ROBOT.force`, `task_frame` is base-relative
xyz/rotation-vector, `wrench` is a task-frame 6D feedforward wrench, and
`tcp_wrench` is a wrench in the **current** TCP frame. Each cycle rotates the TCP
wrench into the task frame, adds the pose-error PD wrench, clips to `max_wrench`,
and applies `selection_vector`. The order is Fx, Fy, Fz, Tx, Ty, Tz (N and Nm).
Defaults use zero feedforward, six selected axes, no automatic sensor zeroing.
Set `stiffness`, `damping`, `limits`, `max_wrench`, `gain_scaling` and
`force_damping` to your validated hardware values before a force experiment;
the example gains are not hardware tuning results. Setting zero feedforward
does not disable pose-error PD. Stopping ends force mode, including feedforward.

## Tasks and operator controls

`TASK = Task()` stops without moving on reset. To customize reset, subclass
`Task` and implement `reset(robot)`. The provided `PegInsertionTask` requires
explicit `home_joints` in radians, retracts in base Z by `retract_distance`, then
moves home. It does not guess a home pose or run automatic randomized resets.
The first Menu start, and the next start after Y/B, invoke reset. Select each
task's mode explicitly in `ROBOT.mode`.

| Control | Effect |
| --- | --- |
| Menu | Start/pause; paused data remain buffered |
| RB | Hold to drive; must first be released after start/reset |
| LB | Precision multiplier while driving |
| X | Toggle base/TCP input frame; ignored in joint modes |
| A | Toggle configured gripper while driving |
| Y | Stop and save the current episode as successful |
| B | Stop and discard the buffered episode |
| Hold View | Stop and exit; discard the unsaved episode |

After Y/B, press Menu to reset and start another episode. Release RB stops
streaming; no new frames are recorded until driving resumes. Thus frame indices
represent sampled driving steps at configured FPS, not wall time across pauses.
A stalled control command, stale robot feedback, safety stop or input failure
ends the session. Reconnect before continuing after a controller fault.
Ctrl-C stops motion and discards the unsaved episode; completed episodes remain.

## Dataset contract

The default state has 18 named physical values: six joint positions (rad), six
base-frame TCP xyz/rotation-vector values (m/rad), and six measured TCP wrench
values (base-frame N/Nm). `include_wrench=False` removes the last six. An optional
gripper adds opening width in metres; configured cameras add image features.
Actions contain the accepted, clipped six physical targets for the chosen mode,
plus optional gripper width. Force mode records the pose equilibrium target,
not measured force or normalized joystick values. The fixed force settings and
task configuration are saved in `ur_setup.json`.

Set `DATASET_ROOT`, `REPO_ID`, `NUM_EPISODES`, `USE_VIDEOS` and `RESUME` in
`config.py`. The writer creates LeRobot v3 metadata/parquet and optional videos.
`RESUME=True` requires the same mode, task/robot settings, FPS and features;
use another directory when changing these. Y is the only episode-save action.
Data are local; the scripts never upload to the Hub. These datasets are not
RLinf ReplayBuffer demos and are not implicitly converted to the old 7/13D state.

## Verification

```bash
cd /workspace/RLinf/external/lerobot
/opt/venv/lerobot/bin/python -m pytest tests/robots/test_ur.py tests/teleoperators/test_xbot.py
```

Tests use SDK fakes and temporary datasets. They do not move hardware or establish
force-mode stability on a real UR.

Task reset runs synchronously: gamepad events are not polled during reset.
Use Ctrl-C or the robot stop controls to interrupt reset motion.

The default mode is `cartesian_force`, selected by `ROBOT.mode` in `config.py`.
Mode dispatch lives in `RTDEController._write` in `src/lerobot/robots/ur/controller.py`.
`ForceConfig.damping=None` derives each host PD damping gain as `2*sqrt(K)`;
changing stiffness also changes the derived damping. This is the critical-damping
approximation for unit effective mass/inertia; in general D=2*sqrt(K*M).
With default K=(1200,1200,1200,100,100,100), D is approximately
(69.282,69.282,69.282,20,20,20). An explicit six-value damping tuple overrides it.
`force_damping=0.3` is the separate native UR forceMode damping parameter.
