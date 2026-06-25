# ATEC 当前 policy 代码整理

这个文件夹整理了两部分内容：

1. 当前 TaskD 正在使用的 locomotion policy：`unitree_b2_rough_dreamwaq`
2. 迁移 Isaaclab_Parkour 思路后的 TaskD crossing 训练代码

其中当前真正接入 `demo/solution.py` 的 loco policy 是 rough DreamWaQ policy；crossing 部分是后续训练“过沟能力”的代码线。

## 1. 当前使用的 loco policy

当前 TaskD 中使用的 locomotion policy 来自：

```text
ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough_dreamwaq/
```

当前整理包中保留了两个可用导出：

```text
ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough_dreamwaq/2026-06-08_17-55-24/exported/policy.pt
ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough_dreamwaq/2026-06-08_18-21-27/exported/policy.pt
```

当前 TaskD 默认使用的是复制到下面位置的 policy：

```text
ATEC2026_Simulation_Challenge/demo/policy.pt
```

这个 policy 对应训练任务：

```text
ATEC-Isaac-Velocity-Rough-Unitree-B2-v0
```

注意：这个不是 TaskDObs policy，也不是 crossing policy。它是 rough locomotion task 通过 DreamWaQ-lite 训练出来的基础行走 policy。

### loco policy 相关代码位置

训练 / 导出入口：

```text
ATEC2026_Simulation_Challenge/scripts/dreamwaq/train.py
ATEC2026_Simulation_Challenge/scripts/dreamwaq/play.py
```

DreamWaQ-lite 算法实现：

```text
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/dreamwaq/dreamwaq_lite.py
```

rough locomotion 环境配置：

```text
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/velocity_env_cfg.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/rough_env_cfg.py
```

PPO 配置：

```text
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py
```

locomotion reward / command / observation / event：

```text
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/
```

TaskD 中加载 loco policy 的代码：

```text
ATEC2026_Simulation_Challenge/demo/solution.py
ATEC2026_Simulation_Challenge/scripts/play_atec_task.py
```

### loco policy 训练命令

单卡训练：

```bash
cd ATEC2026_Simulation_Challenge

python scripts/dreamwaq/train.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000
```

双卡训练：

```bash
cd ATEC2026_Simulation_Challenge

CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/dreamwaq/train.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 \
  --headless \
  --distributed \
  --num_envs 4096 \
  --max_iterations 30000
```

训练日志保存到：

```text
ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough_dreamwaq/
```

### loco policy 导出命令

从默认最新 checkpoint 导出：

```bash
cd ATEC2026_Simulation_Challenge

python scripts/dreamwaq/play.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 \
  --headless \
  --num_envs 64
```

指定 checkpoint 导出：

```bash
cd ATEC2026_Simulation_Challenge

python scripts/dreamwaq/play.py \
  --task ATEC-Isaac-Velocity-Rough-Unitree-B2-v0 \
  --headless \
  --num_envs 64 \
  --checkpoint logs/rsl_rl/unitree_b2_rough_dreamwaq/2026-06-08_17-55-24/model_0.pt
```

导出结果：

```text
logs/rsl_rl/unitree_b2_rough_dreamwaq/<run_name>/exported/policy.pt
```

把导出的 policy 放到 TaskD demo：

```bash
cd ATEC2026_Simulation_Challenge

cp logs/rsl_rl/unitree_b2_rough_dreamwaq/2026-06-08_17-55-24/exported/policy.pt \
  demo/policy.pt
```

### TaskD 中运行当前 loco policy

可视化运行：

```bash
cd ATEC2026_Simulation_Challenge

ATEC_POLICY_MODE=dreamwaq python scripts/play_atec_task.py \
  --task ATEC-TaskD-B2Piper \
  --num_envs 1
```

无界面运行：

```bash
cd ATEC2026_Simulation_Challenge

ATEC_POLICY_MODE=dreamwaq python scripts/play_atec_task.py \
  --task ATEC-TaskD-B2Piper \
  --num_envs 1 \
  --headless
```

打开 debug：

```bash
cd ATEC2026_Simulation_Challenge

ATEC_POLICY_MODE=dreamwaq \
ATEC_DEBUG_SOLUTION=1 \
python scripts/play_atec_task.py \
  --task ATEC-TaskD-B2Piper \
  --num_envs 1
```

## 2. 迁移 Isaaclab_Parkour 的 TaskD crossing 训练代码

crossing 训练目标是只训练“过沟能力”：训练开始时箱子已经在沟壑中，机器人学习沿 waypoint 路线通过沟壑，不训练前面的推箱子阶段。

这部分复用了 Isaaclab_Parkour 的核心思想：

- waypoint / progress reward 引导机器人沿路线前进；
- base height window、near fall、orientation、collision、action smoothness 等 parkour 风格约束；
- teacher / student 配置保留；
- 训练地形使用 TaskD 风格沟壑和沟中箱子；
- policy 接入时仍以 TaskD 可提供的信息为边界，避免依赖部署时拿不到的 privileged 信息。

### crossing 相关代码位置

训练入口：

```text
ATEC2026_Simulation_Challenge/scripts/parkour_taskd/train.py
```

自动 curriculum 训练入口：

```text
ATEC2026_Simulation_Challenge/scripts/parkour_taskd/auto_curriculum_train.py
```

导出入口：

```text
ATEC2026_Simulation_Challenge/scripts/parkour_taskd/export.py
```

RSL-RL wrapper / exporter / runner 依赖：

```text
ATEC2026_Simulation_Challenge/scripts/parkour_taskd/vecenv_wrapper.py
ATEC2026_Simulation_Challenge/scripts/rsl_rl/
```

TaskD crossing 环境配置：

```text
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/crossing_env_cfg.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/crossing.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/rewards.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/terminations.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/terrain.py
```

crossing PPO / teacher / student 配置：

```text
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/parkour_taskd/agents/parkour_rl_cfg.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/parkour_taskd/agents/rsl_teacher_ppo_cfg.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/parkour_taskd/agents/rsl_student_ppo_cfg.py
```

本地查看地形 / smoke test：

```text
ATEC2026_Simulation_Challenge/scripts/view_task_d_crossing.py
ATEC2026_Simulation_Challenge/scripts/smoke_taskd_crossing.py
```

### crossing 任务名

最终地形：

```text
ATEC-TaskD-Crossing-B2Piper
```

curriculum 分阶段地形：

```text
ATEC-TaskD-Crossing-B2Piper-Easy
ATEC-TaskD-Crossing-B2Piper-Mid
ATEC-TaskD-Crossing-B2Piper-Hard
ATEC-TaskD-Crossing-B2Piper
```

### crossing 本地查看地形

查看 easy：

```bash
cd ATEC2026_Simulation_Challenge

python scripts/view_task_d_crossing.py \
  --task ATEC-TaskD-Crossing-B2Piper-Easy \
  --num_envs 1 \
  --enable_cameras
```

查看 mid：

```bash
cd ATEC2026_Simulation_Challenge

python scripts/view_task_d_crossing.py \
  --task ATEC-TaskD-Crossing-B2Piper-Mid \
  --num_envs 1 \
  --enable_cameras
```

查看 final：

```bash
cd ATEC2026_Simulation_Challenge

python scripts/view_task_d_crossing.py \
  --task ATEC-TaskD-Crossing-B2Piper \
  --num_envs 1 \
  --enable_cameras
```

### crossing smoke test

```bash
cd ATEC2026_Simulation_Challenge

python scripts/smoke_taskd_crossing.py \
  --task ATEC-TaskD-Crossing-B2Piper-Easy \
  --num_envs 16 \
  --headless
```

### crossing teacher 训练命令

单卡：

```bash
cd ATEC2026_Simulation_Challenge

CUDA_VISIBLE_DEVICES=3 python scripts/parkour_taskd/train.py \
  --task ATEC-TaskD-Crossing-B2Piper \
  --agent rsl_rl_cfg_entry_point \
  --device cuda \
  --headless \
  --num_envs 2048 \
  --max_iterations 50000
```

双卡：

```bash
cd ATEC2026_Simulation_Challenge

CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/parkour_taskd/train.py \
  --task ATEC-TaskD-Crossing-B2Piper \
  --agent rsl_rl_cfg_entry_point \
  --distributed \
  --device cuda \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
```

训练日志默认保存到：

```text
ATEC2026_Simulation_Challenge/logs/rsl_rl/taskd_crossing_parkour/
```

### crossing 自动 curriculum 训练命令

这个版本会根据终端训练指标中的成功率和跌倒率自动从 easy 切到 mid、hard、final。每次切阶段都会从上一阶段最新 checkpoint 继续训练。

```bash
cd ATEC2026_Simulation_Challenge

python scripts/parkour_taskd/auto_curriculum_train.py \
  --cuda_visible_devices 2,3 \
  --nproc_per_node 2 \
  --num_envs 4096 \
  --experiment_name taskd_crossing_parkour \
  --run_name_prefix auto_curriculum \
  --stage_max_iterations 12000,12000,16000,50000 \
  --stage_min_iterations 3000,3000,4000,8000 \
  --success_thresholds 0.25,0.20,0.12 \
  --fall_max 0.35
```

### crossing 断点继续训练

把 `<run_name>` 和 `<checkpoint>` 换成实际日志目录和模型名：

```bash
cd ATEC2026_Simulation_Challenge

CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/parkour_taskd/train.py \
  --task ATEC-TaskD-Crossing-B2Piper \
  --agent rsl_rl_cfg_entry_point \
  --distributed \
  --device cuda \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --resume \
  --load_run <run_name> \
  --checkpoint <checkpoint>
```

### crossing 导出命令

导出 teacher policy：

```bash
cd ATEC2026_Simulation_Challenge

python scripts/parkour_taskd/export.py \
  --task ATEC-TaskD-Crossing-B2Piper \
  --agent rsl_rl_cfg_entry_point \
  --headless \
  --num_envs 1 \
  --load_run <run_name> \
  --checkpoint <checkpoint>
```

导出结果：

```text
logs/rsl_rl/taskd_crossing_parkour/<run_name>/exported_teacher/policy.pt
logs/rsl_rl/taskd_crossing_parkour/<run_name>/exported_teacher/policy.onnx
```

如果后续训练 student distillation，则 agent 改成：

```text
rsl_rl_student_cfg_entry_point
```

student 导出结果会在：

```text
logs/rsl_rl/taskd_crossing_parkour/<run_name>/exported_deploy/policy.pt
logs/rsl_rl/taskd_crossing_parkour/<run_name>/exported_deploy/policy.onnx
```

### crossing 接入 TaskD 的注意点

crossing policy 现在是单独的“过沟能力”训练线，不等同于当前 `demo/policy.pt` 的基础 loco policy。要接入正式 TaskD 时，需要在 `demo/solution.py` 中做阶段切换：

- 普通行走 / 推箱阶段继续用当前 rough DreamWaQ loco policy；
- 到沟壑附近后切到 crossing policy；
- crossing policy 输入必须保持训练时的 observation 顺序、尺度和 history 语义；
- 不要把 teacher 训练中使用但 TaskD 部署拿不到的信息直接放进最终部署输入。

## 3. 当前整理包保留 / 删除范围

保留：

```text
ATEC2026_Simulation_Challenge/demo/solution.py
ATEC2026_Simulation_Challenge/demo/policy.pt
ATEC2026_Simulation_Challenge/scripts/dreamwaq/
ATEC2026_Simulation_Challenge/scripts/parkour_taskd/
ATEC2026_Simulation_Challenge/scripts/rsl_rl/
ATEC2026_Simulation_Challenge/scripts/play_atec_task.py
ATEC2026_Simulation_Challenge/scripts/view_task_d.py
ATEC2026_Simulation_Challenge/scripts/view_task_d_crossing.py
ATEC2026_Simulation_Challenge/scripts/smoke_taskd_crossing.py
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/dreamwaq/
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/locomotion/
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/parkour_taskd/
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_base/
ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/
ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough_dreamwaq/
```

已从整理包中删除的无关内容：

```text
ACT / TaskE 训练代码
TaskA / TaskB / TaskE 任务代码
TaskDObs / gap / 旧 rough 非当前 policy 日志
demo/solution_act.py
demo/solution_custom_action.py
demo/solution_rl.py
demo/solution_zero.py
__pycache__ / *.pyc / outputs
```
