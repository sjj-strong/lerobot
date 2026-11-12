# LeRobot 原生 UR 与 XBot

这两个示例使用 LeRobot 的 `Robot`、`Teleoperator`、相机工厂和
`LeRobotDataset`，运行时不依赖 RLinf、Ray 或 Pinocchio。RTDE 控制线程默认
500 Hz，遥操作和采集循环默认 30 Hz。自动化测试没有执行真实机械臂运动。

## 环境和执行

在 RLinf 的 UR 容器中使用独立的 Python 3.12 环境：

```bash
cd /workspace/RLinf/external/lerobot
uv pip install --python /opt/venv/lerobot/bin/python --torch-backend cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' -e '.[ur,xbot,robotiq,dataset]'
cd /workspace/RLinf
/opt/venv/lerobot/bin/python external/lerobot/examples/ur/teleoperate.py
/opt/venv/lerobot/bin/python external/lerobot/examples/ur/record.py
```

每次只运行一个脚本。先编辑同目录 `config.py`，填写机械臂 IP、手柄稳定设备路径、
已有的 version-1 evdev 校准 JSON、任务、控制模式、数据目录，并设置
`ENABLE_MOTION=True`。相对路径以当前工作目录为准。
容器需要能访问机器人网络，以及配置的手柄、相机和串口设备。
`/opt/venv/gr00t/bin/python` 是 Python 3.11，不能运行这个 LeRobot 版本。
主机运行前需要将本源码安装到 Python >=3.12 环境；旧 `.venv-ur` 不能直接假定兼容。

## 配置

`URConfig` 支持 `joint_position`、`joint_velocity`、`cartesian_position`、
`cartesian_velocity`、`cartesian_force`，不包含阻抗/directTorque 模式。
关节模式使用左摇杆 Y 控制 `JOINT_INDEX` 指定的关节（0 到 5）。
`TRANSLATION`、`ROTATION`、`JOINT_STEP` 在位置和力控模式中是每帧增量，
在速度模式中分别使用 m/s、rad/s。

`cameras={}` 表示只采集状态，不连接相机。按任务添加任意命名的 LeRobot 相机配置：

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

RealSense 需要安装 `.[intelrealsense]`。`gripper=None` 不连接夹爪，也不添加夹爪数据字段。
配置 Robotiq 后，连接时会执行夹爪激活，按住 RB 时用 A 切换开合。
`payload_mass=None` 保留 UR 控制器已有负载；设置数值时单位为 kg，
`payload_cog` 为相对法兰的重心（m）。不会从历史 RLinf 实验参数推断负载。

力控采用原生 type 2。`ROBOT.force.task_frame` 是相对基座的 xyz/旋转向量，
`wrench` 是任务坐标系六维前馈力，`tcp_wrench` 是**当前 TCP 坐标系**六维前馈力。
每周期旋转 TCP 力到任务坐标系，加上位姿误差 PD，按 `max_wrench` 限幅后应用
`selection_vector`。顺序是 Fx、Fy、Fz、Tx、Ty、Tz，单位为 N、Nm。
默认前馈为零、六轴选中、不自动清零力传感器。
真实力控前，需要在配置中填写经过硬件验证的 `stiffness`、`damping`、`limits`、
`max_wrench`、`gain_scaling` 和 `force_damping`；示例增益不是实际调参结论。
前馈为零不代表没有位姿 PD 力；停止时退出 force mode，同时结束前馈。

## 任务和手柄操作

`TASK = Task()` 复位时只停止，不移动。不同任务继承 `Task`，重写 `reset(robot)`。
提供的 `PegInsertionTask` 要求显式填写弧度制 `home_joints`，先沿基座 Z 轴抬升
`retract_distance`，再返回关节初始位置；不会猜测初始位置或执行随机复位。
第一次按 Menu 开始，以及 Y/B 后再次开始时执行任务复位。
每个任务在 `ROBOT.mode` 中显式选择控制模式。

| 按键 | 行为 |
| --- | --- |
| Menu | 开始/暂停；暂停保留当前数据缓存 |
| RB | 按住才驱动；开始或复位后须先松开一次 |
| LB | 驱动时进入精细控制 |
| X | 切换基座/TCP 输入坐标系；关节模式忽略 |
| A | 驱动时切换已配置夹爪的开合 |
| Y | 停止，将当前 episode 保存为成功数据 |
| B | 停止，丢弃当前 episode |
| 长按 View | 停止退出，丢弃尚未保存的 episode |

Y/B 后按 Menu 复位并开始下一条。松开 RB 停止控制流，也停止添加新帧；
因此数据时间索引表示配置 FPS 下的驱动采样步骤，不包含暂停期间的真实墙钟时间。
命令超时、机器人反馈停滞、安全停止或输入异常会结束本次会话；控制器故障后需要重连。
Ctrl-C 停止运动并丢弃未保存的 episode，已保存数据保留。

## 数据格式

默认状态为 18 维命名物理量：6 维关节位置（rad）、6 维基座系 TCP xyz/旋转向量
（m/rad）、6 维测量 TCP 力/力矩（基座系 N/Nm）。`include_wrench=False` 删除最后六维。
可选夹爪增加米制开口宽度，配置的相机增加图像字段。
动作是对应模式限幅后接受的六维物理目标，以及可选夹爪宽度。
力控记录平衡位姿目标，不把测量力或归一化摇杆值当作执行动作。
固定力控参数和任务配置保存到 `ur_setup.json`。

在 `config.py` 设置 `DATASET_ROOT`、`REPO_ID`、`NUM_EPISODES`、`USE_VIDEOS`、
`RESUME`。输出为 LeRobot v3 元数据、parquet 和可选视频。
续采要求模式、任务/机器人设置、FPS 和特征一致；改变这些配置时使用新目录。
只有 Y 保存 episode。脚本只写本地，不上传 Hub。
这些数据不是 RLinf ReplayBuffer demo，也不会自动转成旧的 7/13 维状态格式。

## 验证

```bash
cd /workspace/RLinf/external/lerobot
/opt/venv/lerobot/bin/python -m pytest tests/robots/test_ur.py tests/teleoperators/test_xbot.py
```

测试使用 SDK 替身和临时数据集，不操作硬件，也不能证明真实 UR 的力控稳定性。

任务复位同步执行，复位期间不读取手柄事件；此时使用 Ctrl-C 或机器人停止控制中断运动。
