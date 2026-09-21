"""在 PushT 任务上评估训练好的 Diffusion Policy checkpoint。

用法
----
    # 官方评估协议（n_test=50），推荐
    python eval_demo.py <checkpoint路径>

    # 指定 episode 数与设备
    python eval_demo.py outputs/.../latest.ckpt --n-test 50 --device cuda:0

    # 只跑几个 episode 快速看一眼（结果不具统计意义，脚本会警告）
    python eval_demo.py <ckpt> --n-test 5

    # 内存受限时分片评估（见下方「分片评估」）
    python eval_demo.py <ckpt> --n-test 10 --n-train 0 --test-seed 100000

内存受限时的分片评估
--------------------
官方协议是 50 个测试 episode，种子为 `test_start_seed + i`（i = 0..49，默认起始 100000）。
`PushTImageRunner` 会为每个 episode 创建一个独立子进程（含 pygame/OpenGL 上下文与
h264 编码器），50 个环境同时运行时内存开销很大，在 16 GB 内存的机器上可能触发 OOM。

本脚本支持把同一批测试集**分成若干片、在独立进程中依次运行**，因为种子是可指定的：

    for s in 100000 100010 100020 100030 100040; do
        python eval_demo.py <ckpt> --n-test 10 --n-train 0 --test-seed $s \
            --output-dir eval_output/shard_$s
    done

这样覆盖的正是**完全相同的 50 个种子**，把 50 个 per-episode 分数合并后，
mean_score 与一次性跑 50 个环境完全一致，而内存峰值降到约 1/5。

说明
----
- 默认**完全沿用 checkpoint 内部保存的配置**（`cfg.task.env_runner`），
  即官方评估协议：`n_test=50`、`max_steps=300`、`test_start_seed=100000`。
- `mean_score` 的定义见 `diffusion_policy/env_runner/pusht_image_runner.py`：
  每个 episode 取 `max_reward`，再对全部 episode 求平均。
  因此 **episode 数直接决定结果可信度**：n_test=1 时该值只能在 0 或 1 附近，
  不能称为「成功率」。

修正记录
--------
本脚本早期版本硬编码 `n_test = 1` 并把 checkpoint 路径写死，
导致「单 episode 结果被当成成功率」以及「路径失效后脚本无法运行」。
现已改为命令行参数 + 官方协议默认值。
"""

import argparse
import pathlib
import sys

import dill
import hydra
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 PushT 上评估 Diffusion Policy checkpoint（默认沿用官方评估协议）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=str, help="checkpoint 文件路径（.ckpt）")
    parser.add_argument("--output-dir", type=str, default="eval_output",
                        help="评估产物输出目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备，例如 cuda:0 或 cpu")
    parser.add_argument("--n-test", type=int, default=None,
                        help="测试 episode 数。留空则沿用 checkpoint 内配置（官方为 50）")
    parser.add_argument("--n-test-vis", type=int, default=None,
                        help="录制视频的 episode 数。留空则沿用配置（官方为 4）")
    parser.add_argument("--n-train", type=int, default=None,
                        help="额外的训练集 rollout 数。评估时可设为 0（不影响测试种子）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="单个 episode 最大步数，留空则沿用配置")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="测试起始种子，留空则沿用配置（官方为 100000）。"
                             "配合 --n-test 可分片评估，见文件末尾说明")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = pathlib.Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        print(f"错误：找不到 checkpoint：{ckpt_path}", file=sys.stderr)
        print("\n提示：训练产物默认写在 outputs/ 下，已被 .gitignore 排除，不随仓库分发。",
              file=sys.stderr)
        print("可先查看：ls outputs/*/*/checkpoints/", file=sys.stderr)
        return 1

    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取 checkpoint，取出训练时保存的完整配置 ──────────────
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    # ── 2. 仅在显式传参时覆盖配置，否则保持官方评估协议不变 ────────
    for key, value in {
        "n_test": args.n_test,
        "n_test_vis": args.n_test_vis,
        "n_train": args.n_train,
        "max_steps": args.max_steps,
        "test_start_seed": args.test_seed,
    }.items():
        if value is not None:
            setattr(cfg.task.env_runner, key, value)

    n_test = cfg.task.env_runner.n_test
    print("评估配置")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  n_test     : {n_test}")
    print(f"  n_test_vis : {cfg.task.env_runner.n_test_vis}")
    print(f"  max_steps  : {cfg.task.env_runner.max_steps}")

    if n_test < 20:
        print(
            f"\n⚠️  警告：n_test={n_test} 过少，mean_score 的统计误差极大，"
            "\n    该结果不能用于与其他方法比较。官方评估使用 n_test=50。\n",
            file=sys.stderr,
        )

    # ── 3. 构建 workspace 并载入权重 ─────────────────────────────
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    policy.eval()
    policy.to(torch.device(args.device))

    # ── 4. 运行环境评估 ─────────────────────────────────────────
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))
    runner_log = env_runner.run(policy)

    # ── 5. 汇总结果 ─────────────────────────────────────────────
    per_episode = [v for k, v in runner_log.items()
                   if k.startswith("test/sim_max_reward_")]

    print("\n=== 评估结果 ===")
    for k, v in runner_log.items():
        print(f"  {k}: {'(视频)' if 'video' in k else v}")

    if "test/mean_score" in runner_log:
        print(f"\n  test/mean_score = {runner_log['test/mean_score']:.4f}")
        if per_episode:
            print(f"  单 episode 范围 : [{min(per_episode):.3f}, "
                  f"{max(per_episode):.3f}]（共 {len(per_episode)} 个）")

    videos = sorted(output_dir.rglob("*.mp4"))
    if videos:
        print(f"\n=== 视频（{len(videos)} 个）===")
        for p in videos[:10]:
            print(f"  {p}")
        if len(videos) > 10:
            print(f"  ... 另有 {len(videos) - 10} 个")

    print("\n提示：报告结果时请同时给出 n_test 与随机种子，单次结果不足以支撑方法间比较。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
