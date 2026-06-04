"""自行检查坐标变换: pixel ↔ world roundtrip"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from config import HEAD_CAM_MATRIX, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX, HEAD_CAM_ROT_MATRIX_INV, IMG_W, IMG_H

robot_pos = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
robot_yaw = np.float32(0.12)
K = HEAD_CAM_MATRIX
ROBOT2CAM = HEAD_CAM_ROT_MATRIX_INV
CAM2ROBOT = HEAD_CAM_ROT_MATRIX
CAM_POS = HEAD_CAM_POS_ROBOT

def p2w(u, v, z):
    cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
    cx, cy_img = (u - K[0,2]) / K[0,0] * z, (v - K[1,2]) / K[1,1] * z
    pr = CAM_POS + CAM2ROBOT @ np.array([cx, cy_img, z], dtype=np.float32)
    return np.array([robot_pos[0] + cy*pr[0] - sy*pr[1],
                     robot_pos[1] + sy*pr[0] + cy*pr[1],
                     robot_pos[2] + pr[2]], dtype=np.float32)

def w2p(pw):
    d = pw - robot_pos
    cys, sys_ = np.cos(-robot_yaw), np.sin(-robot_yaw)
    xr, yr, zr = cys*d[0] - sys_*d[1], sys_*d[0] + cys*d[1], d[2]
    pc = ROBOT2CAM @ np.array([xr - CAM_POS[0], yr - CAM_POS[1], zr - CAM_POS[2]], dtype=np.float32)
    if pc[2] <= 0.05: return None
    ui, vi = int(round(K[0,0]*pc[0]/pc[2] + K[0,2])), int(round(K[1,1]*pc[1]/pc[2] + K[1,2]))
    return (ui, vi) if (0 <= ui < IMG_W and 0 <= vi < IMG_H) else None

print("=== TRANSFORM SELF-TEST ===")
print(f"PITCH={np.rad2deg(np.arcsin(-CAM2ROBOT[0,1])):.1f}°" if abs(CAM2ROBOT[0,1]) > 0.01 else "PITCH: check matrix")
print(f"CAM2ROBOT[0]={CAM2ROBOT[0]}")
print(f"ROBOT2CAM[0]={ROBOT2CAM[0]}")

errors = 0
for u, v, z in [(320, 240, 3.0), (100, 100, 2.0), (500, 400, 5.0), (320, 100, 1.5), (50, 400, 6.0)]:
    w = p2w(u, v, z)
    p = w2p(w)
    status = "✅" if p and abs(p[0]-u) < 2 and abs(p[1]-v) < 2 else "❌"
    if status == "❌": errors += 1
    print(f"  pixel({u:3d},{v:3d}) z={z:.1f}m → world({w[0]:.2f},{w[1]:.2f},{w[2]:.2f}) → pixel{p} {status}")

print(f"\nErrors: {errors}/5")
print("If all ✅: transform is correct, test_offline should work after uploading new files.")
print("If ❌: check that config.py was uploaded with HEAD_CAM_PITCH_DEG = -30.0")