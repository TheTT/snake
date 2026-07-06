# snack

蛇形机器人 MuJoCo 仿真与可视化。

## 依赖

```bash
pip install mujoco imageio[ffmpeg] matplotlib numpy
```

Linux 下需设置环境变量 `MUJOCO_GL=egl`（代码中已通过 `os.environ.setdefault` 自动处理）。

## 可视化模块

基于 MuJoCo 的离屏渲染管线，输出 MP4 视频 + 关节角度曲线图。

### 核心接口 —— 接入自定义关节函数

编辑 `script/joint_functions.py` 中的 `JOINT_FUNCTIONS` 字典（第 266 行）：

```python
JOINT_FUNCTIONS: Dict[str, Callable[[float], float]]
```

**契约**：key 为 MuJoCo 驱动器名（`joint_1_pos` ~ `joint_18_pos`），value 为 `(t: float) -> float`，输入仿真时间（秒），输出关节角度（弧度）。

示例——替换为自己的函数：

```python
def my_joint_1(t):
    return 0.5 * math.sin(2 * math.pi * t / 5.0)

JOINT_FUNCTIONS = {
    "joint_1_pos": my_joint_1,
    # ... 其余关节
}
```

也可以通过调整 `script/gait/sw.json` 中的参数（`ka`、`kb`、`period` 等）改变蛇形步态，而无需修改代码。

### 运行方式

```bash
# 标准模式：单视角追踪，完整物理仿真（重力/接触）
python script/headless_render.py

# FSM 模式：四面板 + 叠加层，无重力/接触，纯运动学展示
python script/headless_render.py --fsm

# 指定输出文件名
python script/headless_render.py --fsm --output demo.mp4
```

输出文件保存在 `res/` 目录下。

### 两种模式对比

| | 标准模式 | FSM（Free-Space Mode） |
|---|---|---|
| 物理 | 有重力/接触/风 | 无重力、无接触、无外力 |
| 视角 | 单相机追踪蛇体中部 | 四面板（主/左/顶/透视） |
| 叠加层 | 无 | 关节连线、轴标记、半透明外壳 |
| 用途 | 观察物理效果 | 观察步态运动学形态 |

### FSM 四面板布局与朝向

```
┌────────────────────┬────────────────────┐
│  主视图 (Main)     │  左视图 (Left)     │
│  沿机器人主轴      │  从机器人左侧      │
│  （朝前方看）      │  （朝右侧看）      │
├────────────────────┼────────────────────┤
│  俯视图 (Top)      │  透视 (Persp)      │
│  从机器人正上方    │  自由视角 +        │
│  往下看            │  关节连线等叠加层  │
└────────────────────┴────────────────────┘
```

相机通过计算全体质心与质量分布主轴，自动追踪机器人朝向：
- **主视图**：镜头看向机器人**前方**（沿质心分布主轴）
- **左视图**：镜头看向机器人**左侧**（主轴的正交左侧方向）
- **俯视图**：镜头从机器人**正上方**俯瞰
- **透视**：固定视角（由配置文件中的 `camera_distance`、`camera_azimuth`、`camera_elevation` 控制），叠加关节连线（蓝/白/黄分段着色）、关节轴标记和半透明外壳

### 配置文件 `script/headless.json`

```json
{
  "duration": 70.0,
  "fps": 60,
  "resolution": "720p",
  "camera_distance": 1.6,
  "camera_azimuth": 180,
  "camera_elevation": -25,
  "fsm": {
    "show_local_axes": false,
    "show_shell_overlay": false
  }
}
```

### 输出产物

- `res/<name>.mp4` — 渲染视频
- `res/<name>_jagl.png` — 关节角度随时间变化曲线（按关节索引 mod 3 分色）

### 文档

各模块详细说明见 `docs/` 目录。关键文件：

| 文件 | 说明 |
|---|---|
| `script/headless_render.py` | 主入口，参数解析与渲染调度 |
| `script/headless_render_passes.py` | 标准与 FSM 两种渲染循环 |
| `script/joint_functions.py` | 关节函数接口（用户需修改处） |
| `script/headless_camera.py` | 追踪相机、主轴估计、FOV 计算 |
| `script/headless_control.py` | 控制分发、自由空间模式设置 |
| `script/headless_joint_utils.py` | 关节连线、轴标记等叠加层绘制 |
| `script/headless_compose.py` | 四面板合成 |
| `script/headless_plot.py` | 关节角度曲线图生成 |
| `script/headless_config.py` | JSON 配置加载与校验 |
