"""
ATEC 2026 Task B 感知层 — 核心 Pipeline
YOLOv11-Nano 检测 + ByteTrack 追踪 + 深度反投影 + 感知输出生成

在 4090 上每帧延迟 ~4ms (YOLO 3ms + ByteTrack 0.5ms + depth 0.3ms + 其他 0.2ms)
"""

import os
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

try:
    import cv2
except Exception:
    cv2 = None

from config import (
    HEAD_CAM, EE_CAM,
    HEAD_CAM_POS_ROBOT, HEAD_CAM_PITCH_RAD,
    HEAD_CAM_ROT_MATRIX, HEAD_CAM_ROT_MATRIX_INV,
    EE_CAM_POS_ROBOT, EE_CAM_ROT_ROBOT,
    ROBOT_INIT_POS, ROBOT_INIT_YAW,
    BIN_CENTER, BIN_RADIUS, BIN_Z_MIN, BIN_Z_MAX,
    CLASS_NAMES, CLASS_NAME_TO_ID, NUM_CLASSES, TOTAL_OBJECTS,
    OBJECT_SIZES, DEFAULT_OBJECT_SIZE,
    GRASP_FIXED_QUAT, DEFAULT_GRASP_FIXED_QUAT, GRASP_DEPTH_OFFSET,
    GRIPPER_HOLDING_MAX_WIDTH,
    TRACK_MAX_AGE, TRACK_MIN_HITS, TRACK_IOU_THRESHOLD,
    YOLO_CONF_THRESHOLD,
    PROPRIO_JOINT_POS_START, PROPRIO_JOINT_POS_LENGTH,
    PROPRIO_FINGER_LEFT, PROPRIO_FINGER_RIGHT,
    PROPRIO_BASE_LIN_VEL, PROPRIO_BASE_ANG_VEL, PROPRIO_PROJECTED_GRAVITY,
    PROPRIO_YAW_FUSION_ALPHA,
)


def quat_multiply(q1, q2):
    """四元数乘法 (w,x,y,z) scalar-first"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_rotate(q, v):
    """用四元数旋转向量 v → q v q*"""
    q_v = np.array([0, v[0], v[1], v[2]], dtype=np.float32)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    rotated = quat_multiply(quat_multiply(q, q_v), q_conj)
    return rotated[1:4]


def rpy_to_quat(roll, pitch, yaw):
    """RPY (rad) → 四元数 (w,x,y,z) scalar-first"""
    return R.from_euler('xyz', [roll, pitch, yaw]).as_quat(scalar_first=True).astype(np.float32)


def quat_to_rpy(q):
    """四元数 (w,x,y,z) → RPY (rad)"""
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')


def yaw_from_gravity(projected_gravity):
    """从投影重力反算机器人 yaw 角
    当机器人水平时 gravity ≈ [0, 0, -1] (归一化后)
    绕 Z 旋转后: g_proj = Rz(yaw)^T @ [0,0,-1]
    => g_x = -sin(yaw), g_y = -cos(yaw)*sin(roll) ≈ -cos(yaw) (roll≈0)
    => yaw = atan2(-g_x, -g_y)
    """
    gx, gy, gz = projected_gravity
    yaw = np.arctan2(-gx, -gy)
    return yaw


def compute_iou(a, b):
    """计算两个 bounding box 的 IoU (均为 [x1, y1, x2, y2] 格式)"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ByteTracker:
    """极简 ByteTrack 纯 Python 实现
    只做 bbox 中心点 + IoU 关联，不依赖外部库
    """

    def __init__(self, max_age=TRACK_MAX_AGE, min_hits=TRACK_MIN_HITS,
                 iou_threshold=TRACK_IOU_THRESHOLD):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = []  # list of dict

    def update(self, detections):
        """
        detections: list of dict, each has keys: 'bbox' [x1,y1,x2,y2], 'class', 'conf'
        Returns: list of dict with added 'track_id', 'bbox', 'class', 'conf'
        """
        # 1. 预测: 所有现有 track age+1
        for t in self.tracks:
            t['age'] += 1

        # 2. 关联（贪心 IoU 匹配）
        matched_track_indices = set()
        matched_det_indices = set()
        assignments = []

        if self.tracks and detections:
            # 构建 IoU 矩阵
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for ti, t in enumerate(self.tracks):
                for di, d in enumerate(detections):
                    iou_matrix[ti, di] = compute_iou(t['bbox'], d['bbox'])

            # 贪心匹配: 按 IoU 降序
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                assignments.append((ti, di))
                matched_track_indices.add(ti)
                matched_det_indices.add(di)
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # 3. 更新已匹配的 track
        for ti, di in assignments:
            t = self.tracks[ti]
            d = detections[di]
            t['bbox'] = d['bbox']
            t['class'] = d['class']
            t['conf'] = d['conf']
            t['age'] = 0
            t['hits'] += 1

        # 4. 为未匹配的检测创建新 track
        for di in range(len(detections)):
            if di not in matched_det_indices:
                self.tracks.append({
                    'track_id': self.next_id,
                    'bbox': detections[di]['bbox'],
                    'class': detections[di]['class'],
                    'conf': detections[di]['conf'],
                    'age': 0,
                    'hits': 1,
                })
                self.next_id += 1

        # 5. 移除过期的 track
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        # 6. 返回已确认的 track（hits >= min_hits）
        return [t for t in self.tracks if t['hits'] >= self.min_hits]


class PerceptionPipeline:
    """
    完整感知 Pipeline:
    1. YOLO 检测 → [cls, conf, bbox]
    2. ByteTrack 追踪 → track_id
    3. 深度反投影 → 3D 世界坐标
    4. 抓取姿态计算
    5. 夹爪状态读取
    6. 桶内判定 + 进度统计
    7. 机器人自定位
    8. 组装 perception_output
    """

    def __init__(self, device='cuda:0', model_path=None):
        """
        Args:
            device: 'cuda:0' 或 'cpu'
            model_path: YOLO 模型路径。None=自动选择微调模型或预训练模型
                        - 若存在 taskb_ycb.pt → 使用微调模型
                        - 否则 → 使用 yolo11n.pt 预训练模型
        """
        self.device = torch.device(device)

        # ---- 自动选择模型 ----
        if model_path is None:
            fine_tuned = os.path.join(os.path.dirname(__file__), 'taskb_ycb.pt')
            if os.path.exists(fine_tuned):
                model_path = fine_tuned
            else:
                model_path = 'yolo11n.pt'  # 自动下载预训练模型

        print(f"[PerceptionPipeline] Loading YOLO: {model_path}")
        self.detector = YOLO(model_path)
        self.detector.to(self.device)
        print(f"[PerceptionPipeline] YOLO loaded OK (device={self.device}).")

        # ---- ByteTrack ----
        self.tracker = ByteTracker()

        self.visualize = os.getenv("ATEC_TASKB_VIS", "0").lower() in {"1", "true", "yes", "on"}
        self.vis_every = max(1, int(os.getenv("ATEC_TASKB_VIS_EVERY", "5")))
        self.vis_dir = os.getenv("ATEC_TASKB_VIS_DIR", os.path.join(os.path.dirname(__file__), "debug_vis"))
        self.vis_show_rgb = os.getenv("ATEC_TASKB_VIS_SHOW_RGB", "1").lower() in {"1", "true", "yes", "on"}
        self.vis_save_yolo = os.getenv("ATEC_TASKB_VIS_SAVE_YOLO", "1").lower() in {"1", "true", "yes", "on"}
        if self.visualize:
            os.makedirs(self.vis_dir, exist_ok=True)
            if cv2 is None:
                print("[TaskB-VIS] cv2 is unavailable; visualization disabled.", flush=True)
                self.visualize = False
            else:
                print(f"[TaskB-VIS] showing head/end-effector RGB and saving YOLO debug images to {self.vis_dir}", flush=True)

        # ---- 相机内参矩阵 (torch) ----
        self.K_head = torch.tensor([
            [HEAD_CAM['fx'], 0, HEAD_CAM['cx']],
            [0, HEAD_CAM['fy'], HEAD_CAM['cy']],
            [0, 0, 1],
        ], dtype=torch.float32, device=self.device)

        self.K_head_inv = torch.inverse(self.K_head)

        self.K_ee = torch.tensor([
            [EE_CAM['fx'], 0, EE_CAM['cx']],
            [0, EE_CAM['fy'], EE_CAM['cy']],
            [0, 0, 1],
        ], dtype=torch.float32, device=self.device)

        self.K_ee_inv = torch.inverse(self.K_ee)

        # ---- end-effector camera 外参旋转矩阵 (预计算自 config.py) ----
        ee_cam_mount = R.from_euler('xyz', EE_CAM_ROT_ROBOT).as_matrix().astype(np.float32)
        ee_cam_axis = np.array([
            [0,  0,  1],
            [1,  0,  0],
            [0, -1,  0],
        ], dtype=np.float32)
        self.ee_cam2robot = ee_cam_mount @ ee_cam_axis
        self.ee_robot2cam = self.ee_cam2robot.T
        self.ee_pos_robot = EE_CAM_POS_ROBOT.copy()

        # ---- 机器人里程计状态 ----
        self.robot_pos = ROBOT_INIT_POS.copy()
        self.robot_yaw = ROBOT_INIT_YAW

        # ---- 物体入桶追踪 ----
        self.in_bin_ids = set()  # 已确认入桶的 track_id 集合

        # ---- 调试用帧计数 ----
        self.frame_count = 0

    def reset(self):
        """每个 episode 开始时调用"""
        self.tracker = ByteTracker()
        self.robot_pos = ROBOT_INIT_POS.copy()
        self.robot_yaw = ROBOT_INIT_YAW
        self.in_bin_ids = set()
        self.frame_count = 0

    # ==========================================================================
    #  Step 1: YOLO 检测
    # ==========================================================================
    def _detect(self, rgb_np):
        """
        rgb_np: (480, 640, 3) uint8 numpy array (end-effector camera)
        Returns: list of dict [{'bbox': [x1,y1,x2,y2], 'class_idx': int, 'conf': float}, ...]
        """
        results = self.detector(rgb_np, conf=YOLO_CONF_THRESHOLD, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
            confs = result.boxes.conf.cpu().numpy()       # (N,)
            classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for box, conf, cls in zip(boxes, confs, classes):
                # 只保留我们关心的 3 类
                if cls in CLASS_NAMES:
                    detections.append({
                        'bbox': box.tolist(),
                        'class': int(cls),
                        'conf': float(conf),
                    })

        return detections

    def _save_debug_visualization(self, rgb_np, detections, tracks, objects_list, target, head_rgb_np=None):
        if not self.visualize or cv2 is None:
            return

        if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
            return

        rgb_bgr = cv2.cvtColor(rgb_np.copy(), cv2.COLOR_RGB2BGR)
        if self.vis_show_rgb:
            cv2.imshow("TaskB end-effector RGB", rgb_bgr)
            if head_rgb_np is not None and head_rgb_np.ndim == 3 and head_rgb_np.shape[2] == 3:
                head_bgr = cv2.cvtColor(head_rgb_np.copy(), cv2.COLOR_RGB2BGR)
                cv2.imshow("TaskB head RGB", head_bgr)
            cv2.waitKey(1)

        if not self.vis_save_yolo or self.frame_count % self.vis_every != 0:
            return

        image = rgb_bgr.copy()
        target_id = None if target is None else target.get('id')
        object_by_id = {obj['id']: obj for obj in objects_list}

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cls_name = CLASS_NAMES.get(det['class'], f"cls_{det['class']}")
            label = f"det {cls_name} {det['conf']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 1)
            cv2.putText(image, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track['bbox']]
            track_id = int(track['track_id'])
            obj = object_by_id.get(track_id)
            is_target = target_id == track_id
            color = (0, 0, 255) if is_target else (0, 220, 0)
            cls_name = CLASS_NAMES.get(track['class'], f"cls_{track['class']}")
            dist = None if obj is None else obj.get('dist_to_robot')
            pos_robot = None if obj is None else obj.get('pos_robot')
            pos_text = ""
            if pos_robot is not None:
                pos_arr = np.asarray(pos_robot, dtype=np.float32)
                pos_text = f" x={pos_arr[0]:.2f} y={pos_arr[1]:.2f}"
            dist_text = "" if dist is None else f" d={float(dist):.2f}"
            label = f"{'TARGET ' if is_target else ''}id={track_id} {cls_name}{dist_text}{pos_text}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if is_target else 1)
            cv2.putText(image, label, (x1, min(image.shape[0] - 6, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        status = f"frame={self.frame_count} detections={len(detections)} tracks={len(tracks)} objects={len(objects_list)} target={target_id}"
        cv2.putText(image, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        filename = os.path.join(self.vis_dir, f"frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(filename, image)

    # ==========================================================================
    #  Step 2: 深度反投影 (像素 → 相机坐标系)
    # ==========================================================================
    def _depth_to_cam(self, depth_np, bbox, cam_cfg=EE_CAM):
        """
        depth_np: (480, 640) float32 深度图 (米)
                  注意: Isaac Lab XYZCamera 输出的深度是沿光轴 (cam_Z) 的距离
        bbox: [x1, y1, x2, y2] 像素坐标
        Returns: 相机坐标系 (OpenCV 约定) 下的 3D 点 (3,) numpy
                 cam_X=右, cam_Y=下, cam_Z=前(光轴/深度方向)
                 若深度无效返回 None
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # clamp 到图像范围
        x1 = max(0, min(x1, cam_cfg['width'] - 1))
        x2 = max(x1 + 1, min(x2, cam_cfg['width']))
        y1 = max(0, min(y1, cam_cfg['height'] - 1))
        y2 = max(y1 + 1, min(y2, cam_cfg['height']))

        # 取 bbox 内 5×5 中心区域的深度中值（抗噪）
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        hs = min(5, (x2 - x1) // 2, (y2 - y1) // 2)
        hs = max(1, hs)

        patch = depth_np[max(0, cy - hs):cy + hs, max(0, cx - hs):cx + hs]
        valid_depths = patch[patch > 0]
        if len(valid_depths) == 0:
            return None

        depth = float(np.median(valid_depths))

        # 针孔反投影: 像素 → OpenCV 相机坐标系
        # cam_X = (u - cx) / fx * depth
        # cam_Y = (v - cy) / fy * depth
        # cam_Z = depth (沿光轴)
        u, v = float(cx), float(cy)
        x_cam = (u - cam_cfg['cx']) / cam_cfg['fx'] * depth
        y_cam = (v - cam_cfg['cy']) / cam_cfg['fy'] * depth
        z_cam = depth

        return np.array([x_cam, y_cam, z_cam], dtype=np.float32)

    # ==========================================================================
    #  Step 3: 坐标变换链 (含俯仰角)
    # ==========================================================================
    def _cam_to_robot(self, p_cam):
        p_robot_offset = self.ee_cam2robot @ p_cam
        return self.ee_pos_robot + p_robot_offset

    def _robot_to_cam(self, p_robot):
        p_robot_offset = p_robot - self.ee_pos_robot
        return self.ee_robot2cam @ p_robot_offset

    def _robot_to_world(self, p_robot):
        """机器人基座坐标系 → 世界坐标系"""
        yaw = self.robot_yaw
        rot_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1],
        ], dtype=np.float32)
        p_world = self.robot_pos + rot_z @ p_robot
        return p_world

    def _world_to_robot(self, p_world):
        """世界坐标系 → 机器人基座坐标系"""
        yaw = self.robot_yaw
        rot_z_inv = np.array([
            [np.cos(yaw),  np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0,             0,          1],
        ], dtype=np.float32)
        p_robot = rot_z_inv @ (p_world - self.robot_pos)
        return p_robot

    # ==========================================================================
    #  Step 4: 抓取姿态
    # ==========================================================================
    def _compute_grasp_quat(self, class_name, obj_pos_world):
        """
        动态计算抓取四元数 (w,x,y,z)
        - 从固定抓取姿态出发
        - 叠加机器人接近方向的 yaw 旋转（使夹爪开口方向对准物体）
        """
        class_name = str(class_name)
        fixed_q = GRASP_FIXED_QUAT.get(class_name, DEFAULT_GRASP_FIXED_QUAT).copy()

        # 机器人 → 物体的方向角
        dx = obj_pos_world[0] - self.robot_pos[0]
        dy = obj_pos_world[1] - self.robot_pos[1]
        approach_yaw = np.arctan2(dy, dx)

        # 绕世界 Z 轴旋转 approach_yaw
        rot_yaw = R.from_euler('z', approach_yaw).as_quat(scalar_first=True).astype(np.float32)
        # grasp = rot_yaw @ fixed_q
        grasp_q = quat_multiply(rot_yaw, fixed_q)

        return grasp_q

    # ==========================================================================
    #  Step 5: 夹爪状态
    # ==========================================================================
    def _read_gripper(self, proprio):
        """
        proprio: (72,) numpy array
        Returns: dict with is_holding, width
        """
        # 从 joint_pos 读取手指位置 (只取 20 维关节位置，不含 joint_vel)
        joint_pos_start = PROPRIO_JOINT_POS_START
        joint_pos_end = PROPRIO_JOINT_POS_START + PROPRIO_JOINT_POS_LENGTH
        joint_pos = proprio[joint_pos_start:joint_pos_end]

        # 手指在 joint_pos 中的局部索引
        finger_left_local = PROPRIO_FINGER_LEFT - PROPRIO_JOINT_POS_START
        finger_right_local = PROPRIO_FINGER_RIGHT - PROPRIO_JOINT_POS_START

        finger_pos = joint_pos[finger_left_local:finger_right_local + 1]
        if len(finger_pos) < 2:
            return {'is_holding': False, 'width': 0.04}

        # 简单近似: 手指开合宽度 = |左右指位置差| * 比例系数
        width = abs(finger_pos[0] - finger_pos[1]) * GRIPPER_HOLDING_MAX_WIDTH * 2.5
        width = np.clip(width, 0.0, 0.08)

        is_holding = 0.002 < width < GRIPPER_HOLDING_MAX_WIDTH

        return {
            'is_holding': bool(is_holding),
            'width': float(width),
        }

    # ==========================================================================
    #  Step 6: 桶内判定
    # ==========================================================================
    def _check_in_bin(self, pos_world):
        """判断物体是否在评分圈内"""
        dist_xy = np.linalg.norm(pos_world[:2] - BIN_CENTER[:2])
        z = pos_world[2]
        return bool(dist_xy <= BIN_RADIUS and BIN_Z_MIN <= z <= BIN_Z_MAX)

    # ==========================================================================
    #  Step 7: 机器人自定位 (里程计积分 + 重力反算 yaw + 角速度融合)
    # ==========================================================================
    def _update_robot_pose(self, proprio, dt=0.02):
        """
        里程计积分 + 重力反算 yaw + 角速度融合
        dt: 控制周期 (Isaac Lab 默认 50Hz → 0.02s, 实际比赛 server.py 为可变)
        """
        lin_vel = proprio[PROPRIO_BASE_LIN_VEL]  # 基座线速度 (机器人坐标系)
        ang_vel = proprio[PROPRIO_BASE_ANG_VEL]  # 基座角速度 [roll, pitch, yaw]
        projected_gravity = proprio[PROPRIO_PROJECTED_GRAVITY]

        # ---- 里程计: 机器人系速度 → 世界系位移 ----
        yaw = self.robot_yaw
        rot_z = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)],
        ])
        lin_vel_world_xy = rot_z @ lin_vel[:2]
        self.robot_pos[0] += lin_vel_world_xy[0] * dt
        self.robot_pos[1] += lin_vel_world_xy[1] * dt
        self.robot_pos[2] = ROBOT_INIT_POS[2]  # 基座高度近似恒定

        # ---- yaw 融合: 重力反算 + 角速度积分 ----
        yaw_from_grav = yaw_from_gravity(projected_gravity)

        # 角速度积分 yaw
        yaw_from_gyro = self.robot_yaw + ang_vel[2] * dt

        # 处理角度缠绕 (确保差值在 [-pi, pi])
        def angle_wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        # 互补滤波: 重力长期稳定 + 角速度短期响应快
        alpha = PROPRIO_YAW_FUSION_ALPHA
        self.robot_yaw = angle_wrap(alpha * yaw_from_grav + (1 - alpha) * yaw_from_gyro)

    # ==========================================================================
    #  主入口
    # ==========================================================================
    def process(self, obs, dt=0.02):
        """
        Args:
            obs: dict, 来自 predicts() 的输入
                {
                    'image': {
                        'ee_rgb':   (1, 480, 640, 3) torch uint8,
                        'ee_depth': (1, 480, 640, 1) torch float32,
                    },
                    'proprio': (1, 72) torch,
                }
            dt: float, 控制周期 (秒)

        Returns:
            perception_output: dict (见 readme/文档)
        """
        self.frame_count += 1

        # ---- 预处理输入 ----
        image_obs = obs.get('image', {})
        head_rgb = image_obs.get('head_rgb')
        
        # 优先使用 ee_rgb/ee_depth，如果不存在则回退到 head_rgb/head_depth
        ee_rgb_key = 'ee_rgb' if 'ee_rgb' in image_obs else 'head_rgb'
        ee_depth_key = 'ee_depth' if 'ee_depth' in image_obs else 'head_depth'
        
        ee_rgb = image_obs[ee_rgb_key].squeeze(0)
        ee_depth = image_obs[ee_depth_key].squeeze(0).squeeze(-1)
        proprio = obs['proprio'].squeeze(0)

        head_rgb_np = None
        if head_rgb is not None:
            head_rgb = head_rgb.squeeze(0)
            if head_rgb.device.type == 'cuda':
                head_rgb_np = head_rgb.cpu().numpy().astype(np.uint8)
            else:
                head_rgb_np = head_rgb.numpy().astype(np.uint8)

        if ee_rgb.device.type == 'cuda':
            ee_rgb_np = ee_rgb.cpu().numpy().astype(np.uint8)
        else:
            ee_rgb_np = ee_rgb.numpy().astype(np.uint8)

        if ee_depth.device.type == 'cuda':
            ee_depth_np = ee_depth.cpu().numpy().astype(np.float32)
        else:
            ee_depth_np = ee_depth.numpy().astype(np.float32)

        if proprio.device.type == 'cuda':
            proprio_np = proprio.cpu().numpy().astype(np.float32)
        else:
            proprio_np = proprio.numpy().astype(np.float32)

        # ---- 更新机器人位姿 ----
        self._update_robot_pose(proprio_np, dt)

        # ---- YOLO 检测 ----
        detections = self._detect(ee_rgb_np)

        # ---- ByteTrack 追踪 ----
        tracks = self.tracker.update(detections)

        # ---- 对每个 track 做深度反投影 → 世界坐标 ----
        objects_list = []  # 所有已追踪物体
        target = None
        best_candidate_dist = float('inf')

        for t in tracks:
            bbox = t['bbox']
            class_idx = t['class']
            track_id = t['track_id']
            conf = t['conf']
            class_name = CLASS_NAMES.get(class_idx, f"cls_{class_idx}")

            # 深度 → 相机坐标 → 机器人坐标 → 世界坐标
            p_cam = self._depth_to_cam(ee_depth_np, bbox, EE_CAM)
            if p_cam is None:
                continue

            p_robot = self._cam_to_robot(p_cam)
            p_world = self._robot_to_world(p_robot)
            p_robot_full = self._world_to_robot(p_world)  # 标准路径获取

            # 抓取点（物体顶面向下偏移）
            grasp_pos_world = p_world.copy()
            grasp_pos_world[2] -= GRASP_DEPTH_OFFSET

            # 抓取姿态
            grasp_q = self._compute_grasp_quat(class_name, p_world)

            # 距离和角度
            dist = float(np.linalg.norm(p_world[:2] - self.robot_pos[:2]))
            yaw_rel = float(np.arctan2(p_world[1] - self.robot_pos[1],
                                        p_world[0] - self.robot_pos[0]) - self.robot_yaw)

            # 桶内判定
            in_bin = self._check_in_bin(p_world)
            if in_bin:
                self.in_bin_ids.add(track_id)

            # 物体尺寸
            size = OBJECT_SIZES.get(class_name, DEFAULT_OBJECT_SIZE)

            obj_info = {
                'id': int(track_id),
                'class': class_name,
                'conf': float(conf),
                'pos_world': p_world.tolist(),
                'pos_robot': p_robot_full.tolist(),
                'grasp_pos_world': grasp_pos_world.tolist(),
                'grasp_quat_world': grasp_q.tolist(),
                'dist_to_robot': dist,
                'yaw_rel': yaw_rel,
                'in_bin': in_bin,
                'size_world': [size['lx'], size['ly'], size['lz']],
                'bbox': bbox,
            }
            objects_list.append(obj_info)

            # 选最近的非桶内物体作为目标
            if not in_bin and dist < best_candidate_dist:
                best_candidate_dist = dist
                target = obj_info

        # ---- 按距离排序 ----
        objects_list.sort(key=lambda x: x['dist_to_robot'])
        self._save_debug_visualization(ee_rgb_np, detections, tracks, objects_list, target, head_rgb_np)

        # ---- 夹爪状态 ----
        gripper = self._read_gripper(proprio_np)
        if target is not None and gripper['is_holding']:
            gripper['held_object_id'] = target['id']
        else:
            gripper['held_object_id'] = None

        # ---- 垃圾桶信息 ----
        bin_dist = float(np.linalg.norm(BIN_CENTER[:2] - self.robot_pos[:2]))
        bin_yaw_rel = float(np.arctan2(BIN_CENTER[1] - self.robot_pos[1],
                                        BIN_CENTER[0] - self.robot_pos[0]) - self.robot_yaw)

        # ---- 进度 ----
        inside_bin = len(self.in_bin_ids)
        remaining = TOTAL_OBJECTS - inside_bin

        # ---- 避障 (其他物体) ----
        obstacles = []
        for obj in objects_list:
            if target is not None and obj['id'] == target['id']:
                continue
            obstacles.append({
                'id': obj['id'],
                'pos_world': obj['pos_world'],
                'radius': 0.25,  # 安全半径
            })

        # ---- 组装输出 ----
        perception_output = {
            'target': target,
            'objects_remaining': [
                {
                    'id': o['id'],
                    'class': o['class'],
                    'dist': o['dist_to_robot'],
                    'pos_world': o['pos_world'],
                    'in_bin': o['in_bin'],
                }
                for o in objects_list
            ],
            'objects_detailed': objects_list,  # 完整信息（给需要更多字段的操作层）
            'gripper': gripper,
            'bin': {
                'center_world': BIN_CENTER.tolist(),
                'radius': BIN_RADIUS,
                'drop_height': BIN_Z_MAX,
                'dist_to_robot': bin_dist,
                'yaw_rel': bin_yaw_rel,
            },
            'obstacles': obstacles,
            'progress': {
                'total': TOTAL_OBJECTS,
                'inside_bin': inside_bin,
                'remaining': remaining,
            },
            'robot': {
                'pos_world': self.robot_pos.tolist(),
                'yaw': float(self.robot_yaw),
            },
        }

        return perception_output