import torch, dill, hydra, pathlib
from omegaconf import OmegaConf

checkpoint = "outputs/2026-09-16/21-05-20/checkpoints/epoch=0050-train_loss=0.025.ckpt"
output_dir = "eval_output"

payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cfg.task.env_runner.n_envs = 1
cfg.task.env_runner.n_train = 0
cfg.task.env_runner.n_test = 1
cfg.task.env_runner.n_test_vis = 1
cfg.task.env_runner.max_steps = 300

cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg, output_dir=output_dir)
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

policy = workspace.model
policy.eval()
policy.to("cuda:0")

env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
runner_log = env_runner.run(policy)

print("=== 评估结果 ===")
for k, v in runner_log.items():
    if "video" in k:
        print(k, ": (视频)")
    else:
        print(k, ":", v)

print("=== 视频文件 ===")
for p in pathlib.Path(output_dir).rglob("*.mp4"):
    print(p)
