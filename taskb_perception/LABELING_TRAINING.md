# Task B 感知：Head 检测 + EE 分割（两套数据、两次训练）

三类物体统一命名（**顺序不能改**）：

| class_id | 名称 |
|----------|------|
| 0 | sugar |
| 1 | mustard |
| 2 | banana |

---

## 一、目录结构（两套分开）

```text
taskb_perception/
├── dataset_head/          ← head_rgb，YOLO **检测**
│   ├── dataset.yaml
│   ├── images/train/      ← manual_collect.py 存的 jpg
│   ├── images/val/
│   ├── labels/train/      ← 你标的 **矩形** txt
│   └── labels/val/
│
├── dataset_ee_seg/        ← ee_rgb 俯视，YOLO **分割**
│   ├── dataset.yaml
│   ├── images/train/      ← manual_collect_ee.py 存的 jpg
│   ├── labels/train/      ← 你标的 **多边形** txt
│   └── depth/train/       ← 可选，推理时用深度
│
└── weights/
    ├── taskb_head_yolo.pt   ← train_yolo.py 输出
    └── taskb_ee_seg.pt      ← train_yolo_seg.py 输出
```

初始化目录：

```bash
python taskb_perception/scripts/init_yolo_datasets.py --which both
```

---

## 二、采集（仿真里只存图）

### Head（导航/远距离发现）

```bash
python taskb_perception/scripts/manual_collect.py \
  --task ATEC-TaskB-B2Piper --enable_cameras \
  --view head --images-only --resume \
  --out taskb_perception/dataset_head
```

### EE 俯视（抓取前精细看物体顶面）

```bash
python taskb_perception/scripts/manual_collect_ee.py \
  --task ATEC-TaskB-B2Piper --enable_cameras \
  --arm-mode preset --resume \
  --out taskb_perception/dataset_ee_seg
```

**不要把 head 和 ee 的图片混在同一个文件夹。**

---

## 三、标注

### 1) Head → YOLO 检测（矩形框）

工具：**LabelImg**（或 Roboflow / CVAT 导出 YOLO detect）

1. 打开 `dataset_head/images/train/`
2. 格式选 **YOLO**
3. 类别：`sugar` `mustard` `banana`
4. 保存目录设为 **`dataset_head/labels/train/`**（与 jpg 同名 `000012.txt`）

**txt 格式（每物体一行，5 个数，归一化 0~1）：**

```text
0 0.512 0.483 0.082 0.095
1 0.731 0.520 0.055 0.130
```

`class cx cy w h`

### 2) EE → YOLO-seg（多边形，算几何中心）

工具推荐（任选）：

- **CVAT** → Export → **Ultralytics YOLO segmentation 1.0**
- **Roboflow** → Segmentation → Export YOLOv8
- **Label Studio** + 多边形 → 导出 YOLO seg

1. 打开 `dataset_ee_seg/images/train/`
2. 用**多边形**沿物体顶面轮廓描（俯视能看到多少描多少）
3. txt 放到 `dataset_ee_seg/labels/train/`

**txt 格式（每物体一行，class + 成对 x y）：**

```text
0 0.41 0.52 0.48 0.51 0.55 0.53 0.50 0.60 0.42 0.58
2 0.20 0.30 0.35 0.28 0.38 0.42 0.22 0.44
```

`class x1 y1 x2 y2 x3 y3 ...`（至少 3 个点）

推理时几何中心：对 mask 或 polygon 求质心 `(cx, cy)`，再用 `ee_depth` 反投影 3D。

---

## 四、检查再训练

```bash
# 检查
python taskb_perception/scripts/check_dataset.py --task head taskb_perception/dataset_head
python taskb_perception/scripts/check_dataset.py --task ee   taskb_perception/dataset_ee_seg

# 训练（分开两次，权重不同）
python taskb_perception/scripts/train_yolo.py \
  --data taskb_perception/dataset_head/dataset.yaml \
  --epochs 80 \
  --out taskb_perception/weights/taskb_head_yolo.pt

python taskb_perception/scripts/train_yolo_seg.py \
  --data taskb_perception/dataset_ee_seg/dataset.yaml \
  --epochs 100 \
  --out taskb_perception/weights/taskb_ee_seg.pt
```

| 模型 | 预训练 | 用途 |
|------|--------|------|
| `yolov8n.pt` | 检测 | head 远距离框 |
| `yolov8n-seg.pt` | 分割 | ee 俯视 mask / 中心点 |

---

## 五、建议数据量

| 数据集 | 建议张数 | 说明 |
|--------|----------|------|
| head detect | 300~600 | 多距离、多角度、可多个框 |
| ee seg | 200~400 | 俯视、物体顶面完整、每类都要 |

可从 150 张先训一版看效果，再分批 `--resume` 补数据。

---

## 六、推理分工（比赛里）

```text
head_rgb  + taskb_head_yolo.pt  → 2D框 + head_depth → 3D 跟踪（导航用）
ee_rgb    + taskb_ee_seg.pt     → mask 质心 + ee_depth → 抓取精定位
```

两个权重**不要混用**：detect 权重不能跑 seg，seg 权重不能代替 head 检测。
