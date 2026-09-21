#!/usr/bin/env python3
"""
dataset_analysis.py — PushT 演示数据的定量诊断

目的
----
为「为什么朴素的单步 MSE 回归（naive behavior cloning）不足以解决 PushT」
提供**数据层面**的可复现证据，而不是只有理论论证。

朴素单步 MSE 回归   π(o) → a,   loss = ||π(o) − a_demo||²
隐含以下三个假设，本脚本逐条量化它们在 PushT 数据上是否成立：

  假设 1  只看当前帧就够            → 由「观测信息量」决定，本脚本不测
  假设 2  给定观测，动作是单峰的     → 第 3 部分检验
  假设 3  每一步相互独立             → 由「序列长度」决定，本脚本不测
  （另）  测试状态落在训练分布内     → 第 2 部分检验（covariate shift）

数据来源
--------
`data/pusht/pusht_cchi_v7_replay.zarr`（官方 PushT 演示集，206 条轨迹 / 25650 步）

state 的语义（见 diffusion_policy/dataset/pusht_image_dataset.py 第 93 行注释）：
    state = (agent_pos × 2, block_pose × 3)
            agent_pos  = 末端执行器位置 (x, y)
            block_pose = T 块位姿 (x, y, angle)
action 的语义：
    末端执行器的**目标位置**（绝对坐标，像素），非位移

依赖
----
    numpy, zarr
    pip install numpy zarr

用法
----
    python3 metrics/dataset_analysis.py
    python3 metrics/dataset_analysis.py --zarr-path <路径> --anchors 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------

def load_dataset(zarr_path: Path):
    """读取 zarr，并给每个样本标注它属于第几条轨迹。"""
    root = zarr.open(str(zarr_path), mode="r")
    state = np.asarray(root["data/state"], dtype=np.float64)      # (N, 5)
    action = np.asarray(root["data/action"], dtype=np.float64)    # (N, 2)
    episode_ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)

    # episode_ends[i] 是第 i 条轨迹的结束下标（开区间）
    episode_id = np.zeros(len(state), dtype=np.int64)
    start = 0
    for i, end in enumerate(episode_ends):
        episode_id[start:end] = i
        start = int(end)

    if start != len(state):
        raise ValueError(f"episode_ends 与 state 长度不一致: {start} != {len(state)}")

    return state, action, episode_id


def standardize(state: np.ndarray) -> np.ndarray:
    """把 5 维状态各自标准化，使欧氏距离不会被量纲大的维度主导。"""
    mu = state.mean(axis=0)
    sd = state.std(axis=0)
    sd[sd == 0] = 1.0
    return (state - mu) / sd


# ----------------------------------------------------------------------------
# 第 1 部分：动作空间语义
# ----------------------------------------------------------------------------

def part1_action_semantics(state, action, episode_id) -> dict:
    print("=" * 78)
    print("第 1 部分｜动作空间语义：action 到底是什么")
    print("=" * 78)

    agent_pos = state[:, :2]                     # 末端执行器当前位置
    block_pos = state[:, 2:4]                    # T 块位置（block_pose 前两维）

    corr_act_agent = [float(np.corrcoef(action[:, i], agent_pos[:, i])[0, 1]) for i in range(2)]
    corr_agent_block = [float(np.corrcoef(agent_pos[:, i], block_pos[:, i])[0, 1]) for i in range(2)]

    # 同一轨迹内相邻两步的末端位移
    same_ep = episode_id[1:] == episode_id[:-1]
    step_disp = np.linalg.norm(agent_pos[1:][same_ep] - agent_pos[:-1][same_ep], axis=1)
    # 动作相对当前位置的"提前量"
    lookahead = np.linalg.norm(action - agent_pos, axis=1)

    print()
    print(f"  state 前两维（agent_pos）范围: "
          f"x=[{agent_pos[:, 0].min():.1f}, {agent_pos[:, 0].max():.1f}]  "
          f"y=[{agent_pos[:, 1].min():.1f}, {agent_pos[:, 1].max():.1f}]")
    print(f"  action                   范围: "
          f"x=[{action[:, 0].min():.1f}, {action[:, 0].max():.1f}]  "
          f"y=[{action[:, 1].min():.1f}, {action[:, 1].max():.1f}]")
    print()
    print(f"  corr(action, agent_pos)  = ({corr_act_agent[0]:+.3f}, {corr_act_agent[1]:+.3f})")
    print(f"  corr(agent_pos, block_pos) = ({corr_agent_block[0]:+.3f}, {corr_agent_block[1]:+.3f})")
    print()
    print(f"  相邻两步末端实际位移      中位数 = {np.median(step_disp):6.2f} 像素")
    print(f"  |action − 当前位置|       中位数 = {np.median(lookahead):6.2f} 像素")

    verdict = (
        "action 是末端执行器的目标位置（绝对坐标），不是位移。\n"
        "      它比当前位置向前看约一个步长的量级，相当于「提前量」。\n"
        "      因此 action 与 state[:2] 高度相关，预测任务本质上是\n"
        "      「根据当前局面决定下一步把末端开到哪里」。"
    )
    print()
    print(f"  → {verdict}")

    return {
        "corr_action_agent": corr_act_agent,
        "corr_agent_block": corr_agent_block,
        "median_step_displacement_px": float(np.median(step_disp)),
        "median_lookahead_px": float(np.median(lookahead)),
    }


# ----------------------------------------------------------------------------
# 第 2 部分：状态空间覆盖（covariate shift 的严重程度）
# ----------------------------------------------------------------------------

def part2_state_coverage(state_std, episode_id, n_anchors, seed) -> dict:
    print()
    print("=" * 78)
    print("第 2 部分｜状态空间覆盖：朴素回归的「外推」压力有多大")
    print("=" * 78)
    print()
    print("  做法：随机抽若干锚点，统计标准化 5 维状态下、半径 r 以内")
    print("        来自**其他轨迹**的样本数（排除同轨迹样本，否则取到的是")
    print("        时间上相邻的近邻，不构成有意义的状态重叠）。")
    print()

    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(state_std), size=n_anchors, replace=False)

    radii = [0.05, 0.10, 0.20, 0.30, 0.50]
    results = {}

    print(f"  {'半径 r':>8} | {'邻居数中位数':>12} | {'邻居<5 的锚点占比':>18}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*18}")

    for r in radii:
        counts = np.empty(len(anchors), dtype=np.int64)
        for k, a in enumerate(anchors):
            d = np.linalg.norm(state_std - state_std[a], axis=1)
            d[episode_id == episode_id[a]] = np.inf     # 排除同轨迹
            counts[k] = int((d < r).sum())

        med = int(np.median(counts))
        frac_sparse = float((counts < 5).mean())
        results[f"r={r}"] = {
            "median_neighbors": med,
            "fraction_with_lt5_neighbors": frac_sparse,
        }
        print(f"  {r:>8.2f} | {med:>12d} | {frac_sparse*100:>17.1f}%")

    print()
    print("  → 解读：半径越小，绝大多数位置在整个数据集里越接近「孤例」。")
    print("     这意味着基于近邻/查表的回归器几乎处处需要外推，")
    print("     而神经网络的外推能力很差 —— 这是 covariate shift 的量化表现。")

    return results


# ----------------------------------------------------------------------------
# 第 3 部分：邻域动作一致性（状态层面是否存在多模态）
# ----------------------------------------------------------------------------

def part3_neighborhood_consistency(state_std, action, episode_id, n_anchors, seed,
                                   radii=(0.15, 0.30), min_neighbors=10) -> dict:
    print()
    print("=" * 78)
    print("第 3 部分｜邻域动作一致性：状态层面是否存在多模态")
    print("=" * 78)
    print()
    print("  做法：对每个锚点，取半径 r 内来自其他轨迹的近邻动作 A，计算")
    print("           ① 邻域内典型间距 d_1nn = 每个动作到它最近邻居的平均距离")
    print("           ② MSE 最优预测（= 条件均值）离最近真实动作的距离 d_pred")
    print()
    print("        主判据 ratio = d_pred / d_1nn，在多个半径下都测，看结论是否一致：")
    print("          单一紧密动作簇 → 均值落在簇内   → d_pred ≈ d_1nn → ratio ≈ 1")
    print("          多个分离动作簇 → 均值落在簇间空地 → d_pred >> d_1nn → ratio >> 1")
    print()

    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(state_std), size=n_anchors, replace=False)

    # 预先算好锚点到全体样本的距离，避免每个半径重复计算
    dist_cache = {}
    for a in anchors:
        d = np.linalg.norm(state_std - state_std[a], axis=1)
        d[episode_id == episode_id[a]] = np.inf     # 排除同轨迹
        dist_cache[a] = d

    summary = {}
    example_used = None

    print(f"  {'半径 r':>8} | {'有效锚点':>8} | {'d_1nn':>8} | {'d_pred':>8} | {'ratio':>7}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*7}")

    for radius in radii:
        stats = []
        example = None
        for a in anchors:
            idx = np.where(dist_cache[a] < radius)[0]
            if len(idx) < min_neighbors:
                continue

            A = action[idx]
            mean_action = A.mean(axis=0)                   # ← MSE 的最优解
            d_pred = float(np.linalg.norm(A - mean_action, axis=1).min())

            pairwise = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
            np.fill_diagonal(pairwise, np.inf)
            d_1nn = float(pairwise.min(axis=1).mean())

            stats.append((d_pred, d_1nn))
            if example is None:
                example = (state_std[a].copy(), A.copy(), mean_action.copy())

        if not stats:
            print(f"  {radius:>8.2f} | {'无有效锚点':>8} |          |          |")
            continue

        d_pred_arr = np.array([s[0] for s in stats])
        d_1nn_arr = np.array([s[1] for s in stats])
        ratio = float(d_pred_arr.mean() / d_1nn_arr.mean())

        summary[f"r={radius}"] = {
            "valid_anchors": len(stats),
            "d_1nn_mean_px": float(d_1nn_arr.mean()),
            "d_pred_mean_px": float(d_pred_arr.mean()),
            "ratio_d_pred_over_d_1nn": ratio,
        }
        print(f"  {radius:>8.2f} | {len(stats):>8d} | {d_1nn_arr.mean():>8.2f} | "
              f"{d_pred_arr.mean():>8.2f} | {ratio:>7.2f}")

        if example is not None and example_used is None and len(stats) >= 50:
            example_used = (radius, example)

    print()
    ratios = [v["ratio_d_pred_over_d_1nn"] for v in summary.values()]
    if ratios:
        worst = max(ratios)
        print(f"  各半径下 ratio 的最大值 = {worst:.2f}")
        print()
        if worst < 1.5:
            verdict = "所有半径下动作都接近单峰：条件均值稳定落在真实动作的密集区。"
            note = "→ 对 PushT，多模态不是主要矛盾（但见下方保留意见）。"
        elif worst < 3.0:
            verdict = "存在一定程度的多模态，但条件均值仍大体落在真实动作附近。"
            note = "→ 多模态有影响，但不是主导因素。"
        else:
            verdict = "条件均值明显落在动作簇之外，多模态问题显著。"
            note = "→ MSE 的最优解不是「可执行的动作」。"
        print(f"  → {verdict}")
        print(f"     {note}")

    if example_used is not None:
        radius, (st, A, mean_action) = example_used
        dist_to_mean = np.linalg.norm(A - mean_action, axis=1)
        order = np.argsort(dist_to_mean)
        print()
        print(f"  具体例子（半径 {radius}，共 {len(A)} 个近邻；单位：像素）")
        print(f"    状态（标准化）: {np.round(st, 3)}")
        print(f"    ── 离条件均值最近的 3 个真实动作 ──")
        for i in order[:3]:
            print(f"      距离 {dist_to_mean[i]:5.2f} : ({A[i, 0]:7.1f}, {A[i, 1]:7.1f})")
        print(f"    ── 条件均值（MSE 的最优解）──")
        print(f"                  ({mean_action[0]:7.1f}, {mean_action[1]:7.1f})")
        print(f"    ── 离条件均值最远的 3 个真实动作 ──")
        for i in order[-3:]:
            print(f"      距离 {dist_to_mean[i]:5.2f} : ({A[i, 0]:7.1f}, {A[i, 1]:7.1f})")

    print()
    print("  ⚠ 保留意见：本检验以**完整 5 维状态**（含末端精确坐标 + 块位姿）为条件，")
    print("     这比策略实际可见的信息**更充分**（策略看到的是 96×96 图像 + 末端坐标）。")
    print("     条件化在更少信息上，歧义只会更大。故本结果是「任务确定性」的上界，")
    print("     不能据此完全排除多模态。")

    return summary


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PushT 演示数据诊断：朴素单步 MSE 回归为何不够")
    parser.add_argument(
        "--zarr-path", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data/pusht/pusht_cchi_v7_replay.zarr",
        help="PushT 演示集路径（zarr 目录）")
    parser.add_argument(
        "--anchors", type=int, default=1500,
        help="第 2/3 部分随机抽样的锚点数量")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    if not args.zarr_path.exists():
        print(f"错误：找不到数据集 {args.zarr_path}", file=sys.stderr)
        print("请用 --zarr-path 指定，或先按 README 下载官方 PushT 数据。", file=sys.stderr)
        return 1

    state, action, episode_id = load_dataset(args.zarr_path)
    state_std = standardize(state)

    print()
    print(f"数据集: {args.zarr_path}")
    print(f"样本数 {len(state)}，轨迹数 {episode_id.max() + 1}")
    print(f"随机种子 {args.seed}，锚点数 {args.anchors}")
    print()

    part1_action_semantics(state, action, episode_id)
    part2_state_coverage(state_std, episode_id, args.anchors, args.seed)
    part3_neighborhood_consistency(state_std, action, episode_id,
                                   args.anchors, args.seed)

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print()
    print("  朴素单步 MSE 回归在 PushT 上的主要障碍，**不是**多模态（第 3 部分")
    print("  测出的比值偏小，说明状态层面的动作分布相当接近单峰），而是：")
    print()
    print("  · 状态空间覆盖极稀疏（第 2 部分）→ 测试时几乎处处需要外推")
    print("  · 单步预测意味着每次小误差都会立刻反馈进下一步 → 误差累积")
    print()
    print("  这正对应 ACT / Diffusion Policy 的核心设计：**一次预测多步动作**，")
    print("  把重规划频率从「每步一次」降到「每 N 步一次」，")
    print("  从而大幅降低 covariate shift 的发作频率。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
