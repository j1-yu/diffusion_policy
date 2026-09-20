# Diffusion Policy 复现：PushT 视觉运动策略

在 PushT 任务上复现论文 **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**（Chi et al., RSS 2023）。

本仓库基于官方开源实现 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) 运行，**策略代码为官方实现，非从零复现**。目标是验证本地 CUDA 环境与评估流程，并复现出与论文接近的结果。

## 结果

| 项目 | 本仓库 | 官方（image PushT, diffusion_policy_cnn） |
|---|---|---|
| 训练轮数 | 100 epoch（loss 1.03 → 0.025） | 3000 epoch |
| 测试成功率 | **93.2%** | `max` **90.7%** / `k_min_train_loss` **83.9%** |
| 随机种子数 | 1 | 3（聚合） |
| 硬件 | RTX 5060 Laptop (8GB), CUDA 12.8, PyTorch 2.8.0 | — |

> ⚠️ **结果的解读边界（重要）**
>
> - **本结果偏高，需要复核。** 官方 3 种子聚合在 **epoch 100 时仅 80.0%**，全程最高 90.7%；本仓库在 **epoch 100 就达到 93.2%**。这个差距需要解释，可能原因：
>   - 评估协议不同（episode 数、最大步数、评估频率）
>   - 单种子方差大（官方单次 checkpoint 曾出现 96.9%，说明单次波动确实很大）
>   - 评估流程存在 bug
> - **不要直接与论文数字比较**。论文数字是 3 种子按 `max` / `k_min_train_loss` 键聚合后的结果，与单次、单种子的评估口径完全不同。
> - 训练轮数也不对等（100 vs 3000 epoch）。
>
> **每个数字都要能说清楚它是怎么来的** —— 这是本仓库下一步要补的功课。

## 本仓库做了什么 / 没做什么

✅ 已完成

- 配置 CUDA 12.8 + PyTorch 2.8.0 环境，跑通官方训练流程
- 在 PushT 上训练 100 epoch，取得 93.2% 的测试成功率（该数字待复核，见上）
- 新增 `verify_pipeline.py`：无需训练即可校验数据加载、模型构建、前向计算与 loss，用于快速排查环境问题

❌ 未包含

- 对策略实现的修改（本仓库使用官方代码）
- 消融实验（去噪步数、预测视野、观测历史等）
- 多随机种子重复与置信区间
- 与官方 3000 epoch 训练水平的对齐验证

## 演示视频

![演示视频](demo.gif)

## 环境配置

Ubuntu 22.04, Python 3.10, CUDA 12.8

```bash
# 1. 安装 mujoco 系统依赖
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# 2. 创建官方 conda 环境（官方推荐 mamba，用 conda 也可）
conda env create -f conda_environment.yaml      # 环境名 robodiff
conda activate robodiff

# 3. 安装本仓库额外依赖
pip install -r requirements.txt

# 4. 快速验证环境（无需训练，约 1 分钟）
python verify_pipeline.py
```

> 修正记录：此前这里写的是 `bash setup_env.sh cuda`，但仓库中并不存在该脚本，已改为官方安装流程并补充了一键验证步骤。

## 训练

训练日志见 train_100.log

## 评估

python eval_demo.py

## 引用

Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023
项目主页 https://diffusion-policy.cs.columbia.edu/
官方实现 https://github.com/real-stanford/diffusion_policy

## 与 ACT 的对比（进行中，暂不作为对比结论）

同一 PushT 任务上另有一个 ACT 实现：https://github.com/j1-yu/act_pusht

⚠️ **当前两者不可比，「93.2% vs 16%」不能说明两种方法的优劣：**

| 问题 | 说明 |
|---|---|
| 训练量不对等 | ACT 训练 5 万步、loss 停留在 0.058；DP 训练 6.7 万步、loss 0.025 |
| 指标异常 | ACT 在 PushT 上仅 16%，这个量级通常意味着训练配置未跑通（训练不足 / 动作归一化 / chunk 与评估协议不匹配），而非方法本身的表现 |
| 评估不严谨 | 两者均为单次评估，无多种子、无置信区间 |

**修正说明**：此前这里直接列出了「DP 93.2% / ACT 16%」的对比表。该数字组合容易让人误读为方法优劣对比，故改为现状说明。

**重做计划**：

1. 修复 ACT 训练配置，目标成功率 ≥ 60%
2. 在相同数据、相同评估协议、相同 episode 数下重新对比
3. 各跑 3 个随机种子，报告均值 ± 标准差，而非单个数字
