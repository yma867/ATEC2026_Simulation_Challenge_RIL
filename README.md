# ATEC2026_Simulation_Challenge_RIL

### 训练命令
# 基础训练命令
python scripts/rsl_rl/train.py --task ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0
# 蹲下训练
python scripts/rsl_rl/train.py --task ATEC-Isaac-Stand-Height-Flat-Unitree-B2-Piper-v0 --headless --video

# 带可视化的训练（需要图形界面）
python scripts/rsl_rl/train.py --task ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0 --video

# 无头模式训练（适合服务器）
python scripts/rsl_rl/train.py --task ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0 --headless --video

# 评估训练好的模型
python scripts/rsl_rl/play.py --task ATEC-Isaac-Velocity-Flat-Unitree-B2-Piper-v0 --enable_cameras

### 测试命令
# 基础运行命令
python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --gt_nav

# 设置导航模式：nearest（找最近目标）
ATEC_TASKB_NAV_MODE=nearest python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --gt_nav

# 设置导航模式：order（按编号顺序 object1-18）
ATEC_TASKB_NAV_MODE=order python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --gt_nav

# 设置导航模式：keyboard（键盘手动控制）
ATEC_TASKB_NAV_MODE=keyboard python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --gt_nav

# 设置停止距离（默认0.5米）
ATEC_TASKB_GT_STOP_DIST=0.6 python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras --gt_nav