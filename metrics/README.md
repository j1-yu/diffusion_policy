# metrics/ — 评估证据

本目录保存可用于**核对 README 中每一个数字**的原始数据。所有数字都可追溯，不是手工填写的。

## 文件说明

| 文件 | 内容 | 大小 |
|---|---|---|
| `eval_history.jsonl` | 每次评估的**完整指标**，包括全部 50 个测试 episode 的原始分数 | 36 KB |
| `train_curve.csv` | 每一轮的 `train_loss`，用于复现学习曲线 | 9 KB |
| `train_config.yaml` | 本次训练使用的完整配置（Hydra 导出，含所有命令行覆盖后的最终值） | 4 KB |

## eval_history.jsonl 的字段

每行是一个 JSON 对象，对应一次评估。关键字段：

```jsonc
{
  "epoch": 650,                    // 第几轮
  "train_loss": 0.0025,            // 该轮训练损失
  "test/mean_score": 0.8996,       // ← README 中报告的主指标
  "test/sim_max_reward_100000": 1.0000,   // 第 1 个测试 episode 的分数
  "test/sim_max_reward_100001": 0.9986,   // 第 2 个
  // ... 共 50 个，种子 100000 ~ 100049
  "test/sim_max_reward_100049": 0.0000,
  "train/mean_score": ...,         // 训练集 rollout 的分数（仅诊断用）
  "test/sim_max_reward_100030": 0.7111     // 等等
}
```

**为什么保留全部 50 个 episode 的分数**：这样才能独立验证均值、计算置信区间、做失败分析。只存一个 `mean_score` 是无法核验的。

## 这些文件是如何生成的

原始日志是 Hydra 的 `JsonLogger` 输出，位于 `data/outputs/<run>/logs.json.txt`。它**每训练一步就写一行**，训练 800 轮后会达到约 12 MB / 11 万行，不适合入库。

因此从中提炼出上述两个文件：

```bash
python3 - <<'PYEOF'
import json, csv
src = "data/outputs/pusht_seed42/logs.json.txt"

# 评估历史：只保留含 test/mean_score 的行（内容逐字保留，未做任何修改）
with open("metrics/eval_history.jsonl", "w") as f:
    for line in open(src):
        d = json.loads(line)
        if d.get("test/mean_score") is not None:
            f.write(line)

# 训练曲线：每轮取最后一次记录的 train_loss
last = {}
for line in open(src):
    d = json.loads(line)
    if "train_loss" in d:
        last[d["epoch"]] = d["train_loss"]
with open("metrics/train_curve.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "train_loss"])
    for e in sorted(last):
        w.writerow([e, f"{last[e]:.6f}"])
PYEOF
```

**注意**：`eval_history.jsonl` 的行是**逐字复制**的，没有重新格式化或截断任何数值。

## 如何核验 README 中的数字

```bash
# 复算最佳 test/mean_score
python3 -c "
import json
rows = [json.loads(l) for l in open('metrics/eval_history.jsonl')]
best = max(rows, key=lambda d: d['test/mean_score'])
print(f\"最佳: epoch {best['epoch']}  mean_score = {best['test/mean_score']:.4f}\")

per = [v for k, v in best.items() if k.startswith('test/sim_max_reward_')]
n = len(per); mean = sum(per)/n
sd = (sum((x-mean)**2 for x in per)/(n-1))**0.5
se = sd/n**0.5
print(f'  n = {n}, std = {sd:.4f}, SE = {se:.4f}')
print(f'  95% CI = [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}]')
print(f'  成功率(coverage>0.95, reward==1.0) = {sum(1 for x in per if x >= 1.0)}/{n}')
"
```

输出应与 README 的「结果」一节一致。

## 未入库的内容

以下内容体积过大，未纳入版本控制（见根目录 `.gitignore`）：

| 内容 | 体积 | 说明 |
|---|---|---|
| `checkpoints/*.ckpt` | 每个约 2.1 GB | 模型权重，可由训练复现 |
| 完整 `logs.json.txt` | 约 12 MB | 逐步训练日志，本目录的文件已涵盖全部结果 |
| `wandb/` 离线日志 | 数十 MB | 含 rollout 视频 |

如需完整原始日志或模型权重，请通过 issue 联系。
