import torch
import hydra
from omegaconf import OmegaConf

cfg = OmegaConf.load("image_pusht_diffusion_policy_cnn.yaml")
OmegaConf.resolve(cfg)

print("=== 1. Loading PushT dataset ===")
dataset = hydra.utils.instantiate(cfg.task.dataset)
print(f"train samples: {len(dataset)}")

print("=== 2. Creating DiffusionUnetHybridImagePolicy ===")
policy = hydra.utils.instantiate(cfg.policy)
policy.eval()
normalizer = dataset.get_normalizer()
policy.set_normalizer(normalizer)
total_params = sum(p.numel() for p in policy.parameters())
print(f"model params: {total_params/1e6:.1f}M")

print("=== 3. Forward + loss ===")
sample = dataset[0]
batch = {
    "obs": {k: v.unsqueeze(0) for k, v in sample["obs"].items()},
    "action": sample["action"].unsqueeze(0),
}

with torch.no_grad():
    loss = policy.compute_loss(batch)
print(f"loss = {loss.item():.4f}")

with torch.no_grad():
    result = policy.predict_action(batch["obs"])
    pred_action = result["action_pred"]
    print(f"predicted action shape = {tuple(pred_action.shape)}")

print("")
print("=== PIPELINE VERIFICATION PASSED ===")
