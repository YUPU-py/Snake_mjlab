# Snake_mjlab - 蛇形机器人速度跟踪训练

基于 IsaacLab 和 Mujoco 的蛇形机器人深度强化学习训练项目，支持 sim2sim 迁移评估。

## 项目概述

本项目使用深度强化学习（PPO 算法）训练蛇形机器人完成速度跟踪任务。蛇形机器人模型代号 **14DOF-DW**（14 个自由度，双被动轮）。

### 核心特性

- 基于 IsaacLab 仿真框架
- 支持Mujoco sim2sim 迁移评估
- 虚拟底盘（Virtual Chassis）控制
- PPO 强化学习算法

## 项目结构

```
Snake_mjlab/
├── snake_mjlab/              # 核心代码
│   ├── env_cfgs.py          # 环境配置
│   ├── rl_cfg.py            # RL 算法配置
│   ├── mdp/                  # MDP 组件
│   │   ├── commands.py       # 命令控制器
│   │   ├── rewards.py        # 奖励函数
│   │   ├── terminations.py  # 终止条件
│   │   ├── observations.py  # 观测函数
│   │   └── virtual_chassis.py # 虚拟底盘计算
│   └── snake_14dof/          # 机器人模型
│       └── xmls/             # Mujoco XML 配置
├── source/                   # IsaacLab 任务注册
├── scripts/                  # 训练脚本
│   ├── run_training.sh      # 启动训练
│   └── run_play.sh          # 可视化播放
├── sim2sim/                 # sim2sim 评估
│   ├── sim2sim_mujoco.py    #Mujoco 可视化
│   └── sim2sim_eval.py      # 评估脚本
└── outputs/                 # 训练输出
```

## 快速开始

### 环境安装

```bash
# 创建 conda 环境
conda create -n snake python=3.11
conda activate snake

# 安装 PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 安装 IsaacLab（请参考官方文档）
# ...

# 安装本项目
cd Snake_Project-main
pip install -e source/snake_project

# 拉取 LFS 资源（如有 USD 模型）
git lfs install
git lfs pull
```

### 训练模型

```bash
# 标准训练
bash scripts/run_training.sh

# 自定义参数训练
python -m mjlab.scripts.train Mjlab-Velocity-Flat-Snake-14DOF --num_envs 4096
```

### 可视化与评估

```bash
# 播放训练好的策略
python -m mjlab.scripts.play \
    Mjlab-Velocity-Flat-Snake-14DOF \
    --checkpoint-file logs/rsl_rl/snake_velocity/xxx/model_xxx.pt \
    --num-envs 1

# sim2sim 评估（需要先导出策略）
python sim2sim/sim2sim_mujoco.py \
    --cmd_vx 0.2 \
    --cmd_vy 0.0 \
    --policy exported/policy.pt
```

### 导出策略

```bash
python -m mjlab.scripts.play \
    Mjlab-Velocity-Flat-Snake-14DOF \
    --checkpoint-file <checkpoint_path> \
    --num-envs 1
```

导出后可在 `logs/rsl_rl/.../exported/` 目录下找到 `policy.pt`。

## 核心配置

### 环境配置 (`snake_mjlab/env_cfgs.py`)

- **观测**: 关节位置/速度、身体角速度、重力方向、速度命令
- **动作**: 7 个偏航关节位置控制
- **奖励**: 速度跟踪奖励 + 相位传播奖励 + 关节幅度奖励

### RL 配置 (`snake_mjlab/rl_cfg.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_dims` | (512, 256, 128) | 网络结构 |
| `entropy_coef` | 0.01 | 熵系数 |
| `learning_rate` | 1e-3 | 学习率 |
| `max_iterations` | 10000 | 最大迭代次数 |

## 调试与开发

### 查看调试指标

训练过程中可在 wandb 中观察以下调试指标：

| 指标名 | 含义 |
|--------|------|
| `debug_world_lin_vel_x/y/z` | base_link 世界坐标速度 |
| `debug_vc_heading_angle` | 虚拟底盘朝向角度 |
| `debug_vc_lin_vel_x/y` | 虚拟底盘速度 |

### 添加自定义奖励

在 `snake_mjlab/mdp/rewards.py` 中添加新的奖励函数：

```python
def my_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    # 计算奖励
    return reward_value
```

然后在 `env_cfgs.py` 中注册：

```python
cfg.rewards["my_reward"] = RewardTermCfg(
    func=my_reward,
    weight=1.0,
    params={"asset_cfg": some_cfg()},
)
```

## 常见问题

### Q: 训练不稳定怎么办？

A: 尝试调整以下参数：
- 增加 `entropy_coef`（如 0.1）
- 降低 `learning_rate`
- 检查奖励函数设计

### Q: sim2sim 迁移效果差？

A: 考虑添加域随机化（domain randomization）或使用更小的速度命令进行训练。

## 参考资料

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Virtual Chassis Paper](https://ieeexplore.ieee.org/document/6094645)

## 许可证

BSD-3-Clause
