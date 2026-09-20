# Diffusion Policy 复现：PushT 视觉运动策略

在 PushT 任务上复现论文 **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**（Chi et al., RSS 2023）。

本仓库基于官方开源实现 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) 运行，**策略代码为官方实现，非从零复现**。目标是在本地 CUDA 环境上跑通官方训练与评估流程。

## 结果

| 项目 | 本仓库 | 官方（image PushT, diffusion_policy_cnn） |
|---|---|---|
| 任务 | PushT（image） | 同 |
| 训练轮数 | 100 epoch（loss 1.03 → 0.025） | 3000 epoch |
| 评估 | **尚未按官方协议完成评估** | `max` 90.7% / `k_min_train_loss` 83.9% |
| 随机种子数 | — | 3（聚合） |
| 硬件 | RTX 5060 Laptop (8GB), CUDA 12.8, PyTorch 2.8.0 | — |

> ⚠️ **关于成功率数字**
>
> 本仓库**目前没有可信的成功率数字**。此前 README 中出现的 93.2% 已移除，原因：
>
> 1. **评估协议不正确**：当时的 `eval_demo.py` 把 `n_test` 设为 **1**（官方为 **50**）。`mean_score` 是全部测试 episode 的均值（见 `pusht_image_runner.py`），只有 1 个 episode 时该值不构成成功率，也无法与其他方法比较。
> 2. **无法追溯**：训练产物在 `outputs/` 下，已被 `.gitignore` 排除，未随仓库分发；仓库中也没有 `eval_log.json` 或 metrics 文件。
> 3. **口径不同**：官方的 90.7% / 83.9% 是 3 个种子按 `max` / `k_min_train_loss` 键聚合后的结果，与单次、单种子的评估不可直接比较。
>
> **待办**：重新训练并按官方协议（`n_test=50`）评估，然后在此处回填「均值 ± 标准差」与 n_test。
>
> 参考：官方在 epoch 100 时 3 种子聚合为 80.0%，全程最高 90.7%。

## 本仓库做了什么 / 没做什么

✅ 已完成

- 配置 CUDA 12.8 + PyTorch 2.8.0 环境，跑通官方训练流程
- 在 PushT 上完成 100 epoch 训练（loss 1.03 → 0.025）
- 新增 `verify_pipeline.py`：无需训练即可校验数据加载、模型构建、前向计算与 loss
- 修正 `eval_demo.py`：改为命令行参数 + 官方评估协议默认值（原版硬编码 `n_test=1` 与失效路径）

❌ 未包含

- 对策略实现的修改（本仓库使用官方代码）
- 可信的评估结果（见上）
- 消融实验（去噪步数、预测视野、观测历史等）
- 多随机种子重复与置信区间
- 模型权重（`.ckpt` 体积大且可由训练复现，未入库）

## 演示视频

![演示视频](demo.gif)

## 环境配置

提供两条**互斥**的安装路径，选一条即可（不要混用）：

### 路径 A：官方 conda 环境（Python 3.9 + torch 1.12.1 + CUDA 11.6）

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
conda env create -f conda_environment.yaml    # 环境名 robodiff
conda activate robodiff
python verify_pipeline.py                     # 校验
```

### 路径 B：本仓库验证通过的环境（Python 3.10 + torch 2.8.0 + CUDA 12.8）

适用于较新的显卡（本项目在 RTX 5060 Laptop / 8GB 上验证）。

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

python verify_pipeline.py                     # 校验
```

> ⚠️ **生成 requirements.txt 时的坑**：如果 shell 里 source 过 ROS 2，
> `/opt/ros/humble` 下的 100+ 个 ROS 包会通过 `PYTHONPATH` 混进 `pip freeze` 的结果里，
> 导致依赖清单完全不可用。正确做法：
>
> ```bash
> env -u PYTHONPATH .venv/bin/python3 -m pip freeze > requirements.txt
> ```

> 修正记录：
> 1. 此前写的是 `bash setup_env.sh cuda`，而仓库中并不存在该脚本；
> 2. 此前的流程会让人先建官方环境（torch 1.12.1）再执行 `pip install -r requirements.txt`（torch 2.8.0），二者互相覆盖。现已拆分为两条互斥路径。

## 训练

```bash
# 单种子（seed 42）
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# 多种子并行（推荐，评估结果才有统计意义）
python ray_train_multirun.py --config-dir=. \
    --config-name=image_pusht_diffusion_policy_cnn.yaml \
    --seeds=42,43,44 --monitor_key=test/mean_score
```

训练产物写入 `outputs/`（单种子）或 `data/outputs/`（多种子），均已被 `.gitignore` 排除。

> 说明：训练脚本的标准输出是 tqdm 进度条，体积可达数 MB 但**不含结构化指标**，因此未纳入版本控制。
> 需要留证据请启用 wandb，或使用 `ray_train_multirun.py`——它会把聚合指标写入 `metrics/logs.json.txt`。

## 评估

```bash
# 默认沿用 checkpoint 内的官方协议：n_test=50, max_steps=300, test_start_seed=100000
python eval_demo.py <checkpoint路径>

# 指定评估规模与设备
python eval_demo.py outputs/.../latest.ckpt --n-test 50 --device cuda:0
```

`n_test` 小于 20 时脚本会警告结果不具统计意义。

也可直接使用官方评估入口（二者协议一致）：

```bash
python eval.py --checkpoint <ckpt> --output_dir data/eval_output --device cuda:0
```

> ⚠️ checkpoint 不会随仓库分发（见 `.gitignore`），需先自行训练或从
> [官方实验日志](https://diffusion-policy.cs.columbia.edu/data/experiments/) 下载。

## 引用

Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023
项目主页 https://diffusion-policy.cs.columbia.edu/
官方实现 https://github.com/real-stanford/diffusion_policy

## 与 ACT 的对比（进行中，暂不作为对比结论）

同一 PushT 任务上另有一个 ACT 实现：https://github.com/j1-yu/act_pusht

⚠️ **当前双方都没有可用于互相比较的结果：**

| 问题 | 说明 |
|---|---|
| DP 侧无有效结果 | 评估协议不正确（`n_test=1` 而非 50），且无存档证据，详见上文「结果」一节 |
| ACT 侧训练不充分 | 训练 5 万步、loss 停留在 0.058（DP 为 6.7 万步、0.025）。16% 这个量级通常意味着训练配置未跑通（训练不足 / 动作归一化 / chunk 与评估协议不匹配），而非方法本身的表现 |
| 评估不严谨 | 双方均为单次评估，无多种子、无置信区间 |

**修正说明**：此前这里直接把「DP 93.2% / ACT 16%」并列成对比表。这两个数字既不可比，DP 侧的数字本身也不可信（见上文），故改为现状说明。

**重做计划**：

1. DP 侧：按官方协议（`n_test=50`）重新评估，最好跑 3 个种子
2. ACT 侧：修复训练配置，目标成功率 ≥ 60%
3. 在相同数据、相同评估协议、相同 episode 数下对比，报告均值 ± 标准差
