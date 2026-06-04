# ATEC 2026 Task B — 感知层（B2-Piper 版）

独立于仿真器的纯 Python 感知方案，可直接拷贝到 AutoDL 4090 上调试和训练。

## 文件结构

```
taskb_perception/
├── config.py                  # 相机参数、场景常量、物体尺寸表（从 Isaac Lab 精确提取）
├── perception_pipeline.py     # 核心 Pipeline（YOLO + ByteTrack + 深度反投影）
├── solution.py                # 比赛入口（复制到 demo/ 即可参赛）
├── generate_synthetic_dataset.py  # 合成训练数据生成器（Headless, 纯 CPU）
├── train_yolo.py              # YOLOv11 微调脚本（GPU 训练）
├── test_offline.py            # 无仿真器离线测试（GT模式 / YOLO模式）
├── check_transform.py         # 坐标变换自检（pixel ↔ world roundtrip）
├── visualize.py               # 离线可视化调试（需已有 RGB+Depth 数据）
├── requirements.txt           # 依赖清单
└── README.md                  # 本文件
```

## 依赖

所有依赖已在你的 `perception` conda 环境中就绪：

```bash
conda activate perception
pip install -r requirements.txt   # 无需操作，所有包已安装
```

关键依赖版本：`ultralytics 8.4.60`、`torch 2.5.1+cu121`、`opencv-python 4.13`

## 感知层输出接口

### perception_output 字典结构 (B2-Piper 版)

```python
perception_output = {
    "target": {
        "id": 5,                                # ByteTrack 追踪 ID
        "class": "mustard_bottle",              # 类别名
        "conf": 0.94,                           # YOLO 检测置信度
        "pos_world": [-8.52, -7.13, 0.15],      # 世界坐标 (x, y, z) 米
        "pos_robot": [1.48, 2.87, -0.45],       # 相对机器人基座坐标
        "grasp_pos_world": [-8.52, -7.13, 0.12], # 抓取点（顶面下移 3cm）
        "grasp_quat_world": [0.0, 0.707, 0.0, 0.707],  # 夹爪姿态 (w, x, y, z)
        "dist_to_robot": 3.25,                  # 到机器人水平距离（米）
        "yaw_rel": 0.78,                        # 相对航向角（弧度）
        "in_bin": False,                        # 是否已在垃圾桶内
        "size_world": [0.22, 0.09, 0.05],       # 物体尺寸 (lx, ly, lz) 米
        "bbox": [120, 80, 200, 160]             # 像素级 BBox [x1, y1, x2, y2]
    },

    "objects_remaining": [                      # 所有待抓取物体（按距离排序）
        {"id": 2, "class": "banana", "dist": 1.20, "pos_world": [...], "in_bin": False},
        {"id": 5, "class": "mustard_bottle", "dist": 3.25, "pos_world": [...], "in_bin": False},
        ...
    ],

    "objects_detailed": [...],                  # 完整信息（含 bbox、grasp_quat、size 等）

    "gripper": {
        "is_holding": False,                    # 是否正在夹持物体
        "width": 0.08,                          # 夹爪当前开合宽度 (m)
        "held_object_id": None                  # 夹持的物体 track_id
    },

    "bin": {
        "center_world": [-3.0, -10.0, 0.0],     # 垃圾桶圆心
        "radius": 1.0,                          # 有效半径 (m)
        "drop_height": 0.50,                    # 投放高度 (m)
        "dist_to_robot": 7.82,                  # 垃圾桶到机器人距离
        "yaw_rel": 0.35                         # 垃圾桶相对方向
    },

    "obstacles": [                              # 避障（其他物体的世界坐标）
        {"id": 3, "pos_world": [...], "radius": 0.25},
        ...
    ],

    "progress": {
        "total": 18,                            # 总物体数
        "inside_bin": 3,                        # 已入桶数
        "remaining": 15                         # 剩余数
    },

    "robot": {
        "pos_world": [-10.0, -10.0, 0.68],      # 机器人基座世界坐标
        "yaw": 0.12                             # 机器人朝向（弧度）
    }
}
```

## 完整工作流程

### 1. 坐标变换自检（必须先跑）
```bash
cd taskb_perception
python check_transform.py
```
预期输出：5/5 ✅ Roundtrip success。
若失败：检查 `config.py` 中 `HEAD_CAM_PITCH_DEG = -30.0` 是否正确上传。

### 2. GT 模式验证坐标变换链精度
```bash
python test_offline.py --gt
```
预期：8 个物体全部检测，平均 3D 误差 < 5cm (🏆 EXCELLENT)。

GT 模式用 Ground Truth bbox 直接走 `深度→cam→robot→world` 坐标变换链，
验证整个反投影流程与正向投影严格互逆。

### 3. 生成合成训练数据
```bash
python generate_synthetic_dataset.py --num_train 2000 --num_val 200
```
纯 CPU 运行（~2-3 分钟 2200 张），输出到 `datasets/` 目录。

### 4. 训练 YOLO（GPU）
```bash
python train_yolo.py --epochs 50 --batch 16
```
4090 上约 10-20 分钟。模型保存到 `runs/train/taskb_ycb/weights/best.pt`。

### 5. 部署模型
```bash
cp runs/train/taskb_ycb/weights/best.pt ./taskb_ycb.pt
```

### 6. YOLO 端到端测试
```bash
python test_offline.py
```
检测 + 追踪 + 定位端到端验证。

### 快速自检
```bash
python solution.py   # 单帧冒烟测试（加载模型 + 跑一帧随机数据）
```

## GPU 内存/模型选择

感知层会自动选择模型：
1. 优先使用 `taskb_ycb.pt`（微调后的模型）
2. 若无，使用 `yolo11n.pt`（自动下载 ~6MB）

yolo11n 显存占用 <500MB，4090 上推理 ~3ms/帧。

如需手动指定模型路径：
```python
pipeline = PerceptionPipeline(device='cuda:0', model_path='path/to/model.pt')
```

## 离线可视化

如果你从仿真器保存了传感器数据（numpy 格式）：

```bash
python visualize.py --rgb head_rgb_frame_123.npy --depth head_depth_frame_123.npy --output annotated.png
```

## 集成到比赛仿真系统

### 交付给队友的文件（4个）
```
demo/
├── solution.py              ← 比赛入口（已实现 get_action_spec + predicts）
├── perception_pipeline.py   ← 核心感知 Pipeline
├── config.py                ← 相机/场景/物体参数配置
└── taskb_ycb.pt             ← YOLO 微调模型
```

### 部署步骤
1. 将上述 4 个文件复制到 `ATEC2026_Simulation_Challenge/demo/` 目录
2. 在 `isaaclab` conda 环境中安装额外依赖（**必须锁 numpy <2.0**）：
   ```bash
   conda activate isaaclab
   pip install "numpy<2.0.0" scipy ultralytics opencv-python
   ```
   > ⚠️ isaaclab 自带 numpy 1.26.x，普通 `pip install scipy` 会触发 numpy 自动升级到 2.x 导致崩溃。
   > `"numpy<2.0.0"` 阻止升级。scipy 1.15+ 与 numpy 1.26 兼容，只有无害的 warning。
3. 启动仿真器：
   ```bash
   python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras
   ```

### 队友需要做的事
`solution.py` 中 `_placeholder_action()` 返回零动作，队友需替换为操控逻辑：
- 输入：`perception_output` 字典（见上方输出接口）
- 输出：20 维 numpy array（12 腿 + 8 臂关节位置）
- 核心字段：`perception_output["target"]["grasp_pos_world"]` + `grasp_quat_world`

## 操作层待办

`solution.py` 中 `_placeholder_action()` 方法返回**零动作**（机器人站立不动）。

你的队友需要在此处接入操作层代码，将 `perception_output` 转换为 B2-Piper 的 20 维关节动作：
- 前 12 维：腿部关节（FR/FL/RR/RL × hip/thigh/calf）
- 后 8 维：臂部关节（arm_joint1-6 + 左指 + 右指）

## 性能

- **YOLO 推理**：~3ms（4090 + TensorRT）
- **ByteTrack 关联**：~0.5ms
- **深度反投影**：~0.3ms
- **其他**：~0.2ms
- **总计**：~4ms/帧（250 FPS），远低于 50Hz 控制周期（20ms）

## 参数来源

所有配置常量从 Isaac Lab v2.3.2 的配置文件精确提取：

| 参数 | 来源文件 | 说明 |
|------|---------|------|
| 相机内参 | `b2.py` UsdCameraCfg | focal_length=24mm, 640×480 |
| 相机外参 | `b2.py` | offset=(0.422, 0.025, 0.062), pitch=30° |
| 垃圾桶位置 | `terrain.py` | VirtualCircle center=(-3, -10) |
| 物体尺寸 | YCB 实物 + USD 模型 | sugar 0.21×0.12×0.07m |
| 物体初始姿态 | `object.py` init_state | rot matrices |
| Proprio 布局 | `b2.py` articulation | 20 关节 72 维观测 |