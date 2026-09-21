# metrics/ — 评估证据

本目录保存可用于**核对 README 中每一个数字**的原始数据。所有数字都可追溯，不是手工填写的。

## 文件说明

| 文件 | 内容 | 大小 |
|---|---|---|
| `eval_history.jsonl` | 每次评估的**完整指标**，包括全部 50 个测试 episode 的原始分数 | 42,028 字节 |
| `train_curve.csv` | 每一轮的 `train_loss`，用于复现学习曲线 | 11,108 字节 |
| `train_full_metrics.csv` | 每一轮的完整指标：`lr`（学习率）、`train_loss`、`train_action_mse_error`、`val_loss`、`train_mean_score` | 46,949 字节 |
| `train_config.yaml` | 本次训练使用的完整配置（Hydra 导出，含所有命令行覆盖后的最终值） | 3,708 字节 |

> `train_full_metrics.csv` 中的后三列每 5 轮才记录一次（`val_every: 5`），
> 因此只有 160 个非空值；`train_mean_score` 每 50 轮记录一次（`rollout_every: 50`），共 16 个非空值。

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

因此从中提炼出上述三个文件（原始日志已在本地删除，不再保留）：

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

# 逐轮指标：按 epoch 归并（每轮有 168 行，后写的覆盖先写的）
rows = {}
for line in open(src):
    d = json.loads(line)
    e = d.get("epoch")
    if e is None:
        continue
    r = rows.setdefault(e, {"epoch": e})
    for src_key, dst_key in [("lr","lr"), ("train_loss","train_loss"),
                             ("train_action_mse_error","train_action_mse_error"),
                             ("val_loss","val_loss"), ("train/mean_score","train_mean_score")]:
        if src_key in d:
            r[dst_key] = d[src_key]

cols = ["epoch","lr","train_loss","train_action_mse_error","val_loss","train_mean_score"]
# 训练曲线：只取 epoch 与 train_loss
with open("metrics/train_curve.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "train_loss"])
    for e in sorted(rows):
        w.writerow([e, f"{rows[e]['train_loss']:.6f}"])
# 完整逐轮指标
with open("metrics/train_full_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for e in sorted(rows):
        w.writerow(rows[e])
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

## 训练产物（已在本地删除，未入库）

训练会产生约 6.3 GB 的产物，体积远超仓库容量，故**未纳入版本控制**，且已在训练完成后从本地删除：

| 内容 | 体积 | 处理 |
|---|---|---|
| 完整 `logs.json.txt` | 14 MB / 134,400 行 | 已删除，本目录文件已涵盖其中全部结果 |
| `wandb/` 离线日志 | 53 MB | 已删除（训练为离线模式，从未同步到云端） |
| `checkpoints/*.ckpt` | 每个 2.04 GB | 见下节 |
| 3 次 OOM 崩溃的存档 | 1.3 MB | 已删除 |

**保留的**：`media/` 下的 rollout 录像（3.8 MB），可作为策略行为的定性证据。

## ⚠️ 关于 checkpoint：本次没有存下最佳模型

这是本次训练的一个**重要缺陷**，在此记录，以免误用。

### 存档机制

存档由 `TopKCheckpointManager` 管理（`diffusion_policy/common/checkpoint_util.py`）。
当已保存的文件数达到 `k` 时，它的判定逻辑是：

```python
if self.mode == 'max':
    if value > min_value:      # 新值比现存最小值大 → 删掉最小值，保留新值
        delete_path = min_path
else:  # 'min'
    if value < max_value:      # 新值比现存最大值小 → 删掉最大值，保留新值
        delete_path = max_path
```

也就是说：**`mode` 必须与指标的「好坏方向」一致**。
`test_mean_score` 越高越好 → 用 `max`；`train_loss` 越低越好 → 应当用 `min`。

### 本仓库的问题

官方 `image_pusht_diffusion_policy_cnn.yaml` 用的是：

```yaml
topk:
  format_str: epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt
  k: 5
  mode: max
  monitor_key: test_mean_score      # 越高越好，配 max ✓
```

但本仓库在最初导入时（commit `7ed08d6`）把它改成了：

```yaml
topk:
  format_str: epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt
  k: 5
  mode: max                         # ✗ train_loss 越低越好，配 max 会保留最差的
  monitor_key: train_loss
```

`train_loss` 在训练中单调下降，所以「数值最高的 k 个」= 训练最早、**最差**的几轮。

### 实际存下的文件

| 文件 | 对应轮次 | 该轮 `test/mean_score` | 评价 |
|---|---|---|---|
| `epoch=0000-train_loss=0.395.ckpt` | epoch 0 | 0.1291 | 最差 |
| `epoch=0050-train_loss=0.030.ckpt` | epoch 50 | 0.6963 | 次差 |
| `latest.ckpt` | epoch 750（按时间戳推断） | 0.8596 | 可用，但非最佳 |

**最佳模型（epoch 650，`mean_score` = 0.8996）从未被保存。**

### 修正

已把配置改回官方写法（`monitor_key: test_mean_score` + `mode: max`），
这样保存的是**评估成绩最好的 k 个模型**。

> 补充约束：`test_mean_score` 只在 rollout 轮次（每 `rollout_every` 轮）才被写入 `step_log`，
> 而 rollout 在 checkpoint 之前执行（`train_diffusion_unet_hybrid_workspace.py` 中 rollout 在
> `for self.epoch` 循环的前段、checkpoint 在后段）。因此 **`checkpoint_every` 必须是
> `rollout_every` 的整数倍**，否则 `get_ckpt_path` 取不到该键会报错。当前两者都是 50，满足条件。

### 影响范围

- ✅ **指标结论不受影响**：评估在训练过程中独立完成，与存档无关，`eval_history.jsonl` 里的数据完整可信
- ❌ 但若要用最佳模型生成演示视频或做后续实验，需按修正后的配置**重新训练**

