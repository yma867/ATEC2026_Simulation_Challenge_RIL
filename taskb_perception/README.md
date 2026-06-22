# Task B 感知层（B2 + Piper）

> **端到端方案（感知 → 导航 → 操作 + Memory Bank 锁定）** 见仓库根目录  
> **[TASKB_PIPELINE.md](../TASKB_PIPELINE.md)**

ATEC 2026 Task B 垃圾捡拾：**感知 / 远距离导航 / 近距离抓取对准** 的 Python 包。

- 仿真联调在 **isaaclab** conda 环境运行
- 本机无仿真时可做离线 YOLO / 单图测试
- 坐标系约定：**robot base 系**（x 前、y 左、z 上，单位米）

---

## 1. 总体方案（三模型 + 两阶段控制）

| 阶段 | 相机 | 臂姿 | 模型 | 输出 | 是否用 depth |
|------|------|------|------|------|--------------|
| **远距离导航** | **ee**（默认，head 暂不启用） | stowed（默认） | `ee_det` | 2D 框 → **首帧锁定** → 光轴对准 → 直行 | 可选（仅停车距离） |
| **近距离抓取对准** | ee | overhead（垂直俯视） | `ee_seg` | mask 质心 → 对齐图像中心 | **否**（纯视觉伺服） |
| **3D 跟踪 / 抓取点**（可选） | head + ee | — | detect + depth 反算 | `pos_b` + 跟踪 EMA | 是 |

### 1.1 远距离导航（当前默认）：光轴中线 + Memory Bank 锁定

**不再依赖 depth 反算 base 3D**（外参/深度误差较大），改为 **EE 单视角 + 首帧锁定 + Memory Bank 跨帧关联**：

```text
帧0: ee_det → 按 conf 排序 → 锁最高框 → 写入 Memory Bank.active
帧N: ee_det → 用 active 预测位置 (last_u, last_v, last_bbox) 关联本帧检出
     → 在范围内则认定为同一目标 → 更新 active
     → 光轴对准 err_u → 转向 → 直行
     → depth 到达 → complete → 锁下一个
```

- **Memory Bank**：`active`（当前锁）+ `completed`（已处理历史，防重复锁）
- **head 默认关闭**（`nav_detect_source="ee"`）
- depth 仅用于 **接近时减速停车**

详见 [TASKB_PIPELINE.md §4 Memory Bank](../TASKB_PIPELINE.md#4-memory-bank-目标锁定核心)

### 1.2 近距离抓取：EE 分割视觉伺服

臂切换到 **overhead**（垂直俯视）后：

```text
ee_seg → mask 几何质心 (u, v)
    → 与标定点 (u*, v*) 对齐（默认图像中心）
    → 对准后锁 XY，仅 Z 下降抓取
```

### 1.3 旧方案（备用）：depth 反算 pos_b

```text
2D 框 + ee_depth/head_depth → 内参 → 外参 → pos_b (base)
```

仍保留在 `ee_det3d.py`，可通过 `--nav-mode pos_b_3d` 或 `nav_method="pos_b_3d"` 启用。  
可用 `gt_eval.py` 与仿真 GT 对比误差。

---

## 2. 模型权重

路径均相对 **仓库根目录** `ATEC2026_Simulation_Challenge/`：

| 变量 | 路径 | 用途 |
|------|------|------|
| `head_det` | `taskb_perception/train_fast/train_fast/head_det/weights/best.pt` | 头相机远距离 detect |
| `ee_det` | `taskb_perception/train_fast/train_fast/ee_det/weights/best.pt` | EE 相机远距离 detect |
| `ee_seg` | `taskb_perception/ee_seg/ee_seg/weights/best.pt` | EE 垂直俯视 seg |

类别：`0=sugar`, `1=mustard`, `2=banana`

默认配置见 `taskb_perception/config.py` → `PerceptionConfig`。

---

## 3. 目录结构

```text
taskb_perception/
├── README.md                          # 本文档
├── LABELING_TRAINING.md               # 标注与训练补充说明
├── requirements.txt
├── taskb_perception/                  # Python 包
│   ├── config.py                      # 相机内参、导航/对准参数、权重路径
│   ├── module.py                      # PerceptionModule 主入口
│   ├── collector_nav.py               # B2 步态驱动 + AxisNavController（光轴导航）
│   ├── axis_nav.py                    # 兼容转发 → collector_nav
│   ├── detector.py                    # head YOLO detect
│   ├── ee_det3d.py                    # ee detect + depth → pos_b
│   ├── ee_seg_align.py                # ee seg 质心对准
│   ├── ee_collect_utils.py            # 臂姿态 preset、手调关节、相机视角工具
│   ├── gt_eval.py                     # 反算 pos_b vs 仿真 GT 误差评估
│   ├── tracker.py                     # 3D 多目标跟踪
│   ├── grasp_pose.py                  # 抓取点/姿态
│   ├── math3d.py / obs_utils.py       # 坐标变换、obs 解析
│   └── types.py
├── scripts/
│   ├── ee_det_nav.py                  # ★ 仿真演示：光轴导航 + 可选 pos_b_3d
│   ├── ee_rgbd_snapshot.py            # EE RGBD 快照、手调臂角、detect 可视化
│   ├── ee_det_eval_gt.py              # 批量 GT 误差评估（headless）
│   ├── manual_collect.py              # head 相机 WASD 采集
│   ├── manual_collect_ee.py           # EE 俯视 seg 数据集采集
│   ├── test_ee_seg_center.py          # 离线 seg 质心测试
│   ├── train_yolo.py / train_yolo_seg.py
│   └── ...
└── integration/
    ├── solution_ee_det_nav.py         # 光轴导航 AlgSolution 示例
    └── solution_perception_hook.py    # PerceptionModule 全链路示例
```

---

## 4. 环境依赖

| 包 | 版本 |
|----|------|
| Python | 3.11 |
| numpy | ≥ 1.24, **< 2.0** |
| ultralytics | ≥ 8.0 |
| opencv-python, scipy | — |
| torch | isaaclab 环境自带，勿覆盖 |

```bash
conda activate isaaclab
pip install "numpy<2.0.0" scipy ultralytics opencv-python
```

---

## 5. 机械臂姿态

| `--arm-pose` | 关节 preset | 用途 |
|--------------|-------------|------|
| **stowed**（默认） | `B2_PIPER_STOWED_JOINTS` | 臂收在狗身，EE/head 朝前看场景，**远距离导航** |
| overhead | 前伸俯视 | **抓取 / ee_seg 对准** |
| zero | 全 0 | 调试 |

手调关节（Isaac 窗口按键）：

| 键 | 作用 |
|----|------|
| I/K | joint2 抬高/放低 |
| J/L | joint3 收/伸 |
| U/O | joint5 腕俯仰 |
| Z/X | joint1 底座旋转 |
| [ / ] | joint4 |
| ; / ' | joint6 |
| B | 细调模式 |
| **Y** | 打印并保存关节角 → `snapshots/ee_rgbd/arm_joints_tuned.json` |

加载已保存角度：

```bash
python taskb_perception/scripts/ee_rgbd_snapshot.py \
  --arm-joints snapshots/ee_rgbd/arm_joints_tuned.json \
  --task ATEC-TaskB-B2Piper --enable_cameras
```

---

## 6. 仿真快速开始

```bash
conda activate isaaclab
cd ATEC2026_Simulation_Challenge-main

# ★ 光轴中线导航演示（默认）
python taskb_perception/scripts/ee_det_nav.py \
  --task ATEC-TaskB-B2Piper --enable_cameras \
  --arm-pose stowed --nav-mode axis_align
```

| 按键 | 功能 |
|------|------|
| **N** | 开关自动导航（开启时首帧锁定最高 conf 目标） |
| **C** | 手动标记当前目标已完成，锁定下一个 |
| **F** | 打印全部检出 + 当前 LOCK 状态 |
| W/S/A/D | 手动移狗 |
| I/K/J/L… | 手调臂角 |
| Y | 保存臂关节角 |
| Q | 退出 |

OpenCV 窗口说明：

- **浅蓝框** = YOLO 检出的全部目标
- **洋红粗框 `LOCK#N`** = 当前锁死跟踪的导航目标
- **绿色竖线** = 光轴中心线（u = cx）
- 左上角 `det=N` = 本帧检出数量

### 6.1 EE RGBD 调试

```bash
python taskb_perception/scripts/ee_rgbd_snapshot.py \
  --task ATEC-TaskB-B2Piper --enable_cameras --arm-pose stowed
```

### 6.2 旧方案：pos_b 反算 + GT 评估

```bash
python taskb_perception/scripts/ee_det_nav.py \
  --nav-mode pos_b_3d --gt-eval

python taskb_perception/scripts/ee_det_eval_gt.py \
  --task ATEC-TaskB-B2Piper --enable_cameras --headless
```

---

## 7. 核心 API

### 7.1 导航队友（推荐：EE 光轴 + 目标锁）

```python
from taskb_perception import AxisNavController, PerceptionConfig

cfg = PerceptionConfig()  # nav_detect_source 默认 "ee", nav_lock_enabled=True
nav = AxisNavController(cfg)

bank = nav.memory_bank              # active + completed
vel = nav.compute_velocity(obs)
target = nav.nav_target             # locked=True, lock_id=N
nav.complete_current()              # active → completed，锁下一个
```

或通过 `PerceptionModule`：

```python
from taskb_perception import PerceptionModule, PerceptionConfig

p = PerceptionModule(PerceptionConfig())
vel = p.compute_nav_velocity(obs)
target = p.last_axis_nav_target
done = p.nav_completed_count

p.complete_nav_target()  # 抓取完成 → 下一个
```

参考：`integration/solution_ee_det_nav.py`

### 7.2 抓取对准队友（EE seg）

```python
from taskb_perception import EESegAligner, PerceptionConfig
from taskb_perception.obs_utils import parse_rgb

aligner = EESegAligner(PerceptionConfig())
ee_rgb = parse_rgb(obs["image"], "ee_rgb")
center = aligner.best_center(ee_rgb)  # EESegCenter | None

if center and not center.aligned:
    err_u, err_v = center.err_u, center.err_v  # 驱动臂/狗微调
# 对准后 → 仅 Z 下降
```

### 7.3 3D 跟踪 / 抓取点（全链路）

```python
from taskb_perception import PerceptionModule, PerceptionConfig

p = PerceptionModule(PerceptionConfig())
out = p.update(obs)

if out.next_target:
    pos_b = out.next_target.pos_b
    dist = out.next_target.distance_xy()

# 抓取前锁定，防抖动
locked = p.lock_target(out.next_target.track_id)
grasp_pos = locked.grasp_pos_b
```

参考：`integration/solution_perception_hook.py`

### 7.4 EE detect → pos_b（单帧，旧导航）

```python
from taskb_perception import EEDetector, PerceptionConfig
from taskb_perception.obs_utils import parse_rgb, parse_depth

det = EEDetector(PerceptionConfig())
dets = det.detect_all(
    parse_rgb(obs["image"], "ee_rgb"),
    parse_depth(obs["image"], "ee_depth"),
)
for d in dets:
    print(d.obj_class, d.pos_b, d.depth_m)
```

---

## 8. 配置说明（`PerceptionConfig`）

### 检测

| 参数 | 默认 | 说明 |
|------|------|------|
| `nav_detect_source` | `"ee"` | `"ee"` / `"head"` / `"both"`（默认仅 ee） |
| `conf_threshold` | `0.35` | YOLO 置信度阈值 |
| `enable_ee_nav_detect` | `True` | 是否加载 ee_det |

### EE 目标锁（`nav_lock_enabled=True`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `nav_lock_enabled` | `True` | 首帧 conf 锁死 + 帧间跟踪 |
| `nav_lock_auto_acquire` | `True` | `compute_velocity` 时自动锁 |
| `nav_lock_auto_next` | `True` | depth 到达后自动 `complete_current` |
| `nav_lock_match_max_dist_px` | `140` | 帧间质心最大跳变 |
| `nav_lock_max_missed` | `45` | 连续丢失帧后放弃当前锁 |
| `nav_lock_exclude_dist_px` | `48` | 已完成目标排除半径 |

### 光轴导航（`nav_method="axis_align"`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `nav_axis_target_u/v` | 320 / 240 | 光轴在图像上的对准点 |
| `nav_axis_tol_u_px` | `12` | 认为「在中线上」的 u 容差（像素） |
| `nav_axis_turn_gain` | `0.022` | 转向增益 |
| `nav_axis_forward_vx` | `0.45` | 对准后前进速度 |
| `nav_axis_align_before_forward` | `True` | 先对准再前进 |
| `nav_axis_arrive_depth_m` | `1.25` | depth 停车距离（米） |

### EE seg 对准

| 参数 | 默认 | 说明 |
|------|------|------|
| `ee_align_target_u/v` | 320 / 240 | 质心应对齐的像素点 |
| `ee_align_tol_px` | `8` | 对准容差 |

---

## 9. 推荐状态机（与队友分工）

```text
SEARCH / APPROACH     臂 stowed + ee_det + 光轴锁定导航（一次一个）
        ↓  depth 到达 / complete_current()
GRASP_ALIGN           臂 overhead + ee_seg 质心 ↔ 图像中心
        ↓  aligned
DESCEND / GRASP       锁 XY，仅 Z 下降 + 抓取队友
        ↓
PLACE                 导航至 TARGET_BIN_XY (-3, -10)
```

---

## 10. 相机与内参

| 相机 | fx ≈ | cx, cy | 用途 |
|------|------|--------|------|
| head | 732.5 | 320, 240 | 远距离、视野宽 |
| ee | 458 | 320, 240 | 臂载相机，随 stowed/overhead 变 |

depth 来自仿真 `obs["image"]["head_depth"]` / `ee_depth`，单位 **米**，为传感器真实读数。

---

## 11. 数据流示意

```text
                    ┌─────────────────────────────────────┐
                    │           远距离导航 (默认)            │
  ee_rgb  ──► ee_det ──► 首帧 LOCK(最高conf) ──► 帧间跟踪
                          │
                          └──► AxisNavController ──► [vx,vy,wz]
                                    └─► nav_target (LOCK#N)

  head_rgb ──► head_det ──► 默认不启用 (nav_detect_source="ee")

                    ┌─────────────────────────────────────┐
                    │         近距离抓取对准                 │
  ee_rgb  ──► ee_seg ──► 质心 (u,v) ──► err vs (u*,v*)

                    ┌─────────────────────────────────────┐
                    │      3D 跟踪 / 抓取 (可选)            │
  detect + depth ──► pos_b ──► Tracker3D ──► grasp_pos_b
```

---

## 12. 训练与采集（摘要）

详细见 `LABELING_TRAINING.md`。

```bash
# head detect 采集
python taskb_perception/scripts/manual_collect.py \
  --task ATEC-TaskB-B2Piper --enable_cameras \
  --out taskb_perception/dataset_head

# EE 俯视 seg 采集
python taskb_perception/scripts/manual_collect_ee.py \
  --task ATEC-TaskB-B2Piper --enable_cameras \
  --out taskb_perception/dataset_ee_seg

# 训练
python taskb_perception/scripts/train_yolo.py \
  --data taskb_perception/dataset_head/dataset.yaml --epochs 80

python taskb_perception/scripts/train_yolo_seg.py \
  --data taskb_perception/dataset_ee_seg/dataset.yaml --epochs 100
```

---

## 13. 常见问题

**Q: 为什么只跟踪一个框？其他框会干扰吗？**  
A: 开启 `nav_lock_enabled` 后，**首帧按置信度锁死**一个 EE 框，后续帧用 bbox/质心匹配同一目标，不会被其他高 conf 框抢走。按 **C** 或到达 depth 后处理下一个。

**Q: 为什么不用 head？**  
A: head/ee 跨视角难以稳定对应同一物体 ID，默认 `nav_detect_source="ee"`。需要时可改 `"both"` 或 `"head"`。

**Q: 为什么只看到一个框？**  
A: 导航只跟 `nav_target` 走，但 YOLO 可能检出多个。按 **F** 看 `YOLO 检出 N 个`；画面浅蓝框为全部检出。同时视野里通常只有 1～2 个物体。

**Q: 画面/场地在抖？**  
A: Isaac 窗口绑定了 **ee_camera**（随狗/臂动），不是地面在滑。可改看 OpenCV 窗口或切换 World 视角。

**Q: `ModuleNotFoundError: axis_nav`？**  
A: 光轴导航在 `collector_nav.py` 里，确保同步最新代码；`axis_nav.py` 仅为兼容转发。

**Q: 转向方向反了？**  
A: 将 `nav_axis_turn_gain` 改为负值，或在代码里翻转 wz 符号。

**Q: stowed 视角不对？**  
A: 用 `ee_rgbd_snapshot.py` 手调关节，按 **Y** 保存，用 `--arm-joints` 加载。

**Q: pos_b 反算误差大？**  
A: 正常，因此默认改用光轴导航；若必须用 pos_b，用 `gt_eval.py` 对比并校准 `ee_cam` 外参或改用 live 外参。

**Q: pip 后 isaaclab 崩了？**  
A: 勿 `pip install -r requirements.txt` 覆盖 torch。只装：`pip install "numpy<2.0.0" scipy ultralytics opencv-python`

---

## 14. 集成 checklist

- [ ] 三个权重文件路径正确
- [ ] 导航：`AxisNavController` + `B2LocomotionDriver`（`demo/policy.pt`）
- [ ] Approach 阶段：`arm_pose=stowed`，`nav_detect_source=ee`，`nav_lock_enabled=True`
- [ ] 抓取完成后调用 `complete_current()` 再导航下一个
- [ ] Grasp 阶段：切 `arm_pose=overhead`，启用 `EESegAligner`
- [ ] 抓取前 `lock_target()` 防抖动
- [ ] 放置目标：`TARGET_BIN_XY = (-3.0, -10.0)`

---

## 15. 版本说明

| 脚本 | 版本标识 | 说明 |
|------|----------|------|
| `ee_det_nav.py` | v6 | EE 光轴导航 + 目标锁，C=下一个 |
| `ee_rgbd_snapshot.py` | v5 | stowed + 手调臂 |
| 导航核心 | `collector_nav.AxisNavController` | 光轴中线方案 |

如有问题，优先在仿真里跑 `ee_det_nav.py` + 按 **F** 打印诊断信息。
