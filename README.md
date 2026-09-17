# Diffusion Policy 复现：PushT 视觉运动策略

复现论文 Diffusion Policy (RSS 2023)，在 PushT 任务上训练视觉运动策略。

## 结果

训练 100 epoch，loss 从 1.03 降到 0.025
测试成功率 93.2%
硬件 RTX 5060 Laptop (8GB), CUDA 12.8, PyTorch 2.8.0

## 演示视频

demo.mp4

## 环境配置

Ubuntu 22.04, Python 3.10
安装：bash setup_env.sh cuda
依赖：requirements.txt

## 训练

训练日志见 train_100.log

## 评估

python eval_demo.py

## 引用

Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023
项目主页 https://diffusion-policy.cs.columbia.edu/
