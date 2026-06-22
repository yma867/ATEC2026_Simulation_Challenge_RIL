"""Task B 物体类别与 3D 尺寸（用于仿真自动标注 2D bbox）。"""

from __future__ import annotations

# YOLO class_id
CLASS_SUGAR = 0
CLASS_MUSTARD = 1
CLASS_BANANA = 2

CLASS_NAMES = ["sugar", "mustard", "banana"]

# object_i 的类别（Task B env_cfg：1-6 sugar, 7-12 mustard, 13-18 banana）
def object_index_to_class(obj_idx: int) -> int:
    if obj_idx <= 6:
        return CLASS_SUGAR
    if obj_idx <= 12:
        return CLASS_MUSTARD
    return CLASS_BANANA


# 物体局部坐标系下的半尺寸 (half_x, half_y, half_z)，单位 m
# 用于把 3D 真值投影成 2D bbox，不需要人工标注
OBJ_HALF_EXTENTS_3D: dict[int, tuple[float, float, float]] = {
    CLASS_SUGAR: (0.050, 0.044, 0.030),
    CLASS_MUSTARD: (0.030, 0.030, 0.080),
    CLASS_BANANA: (0.100, 0.040, 0.020),
}

NUM_OBJECTS = 18
