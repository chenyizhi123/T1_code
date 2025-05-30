import torch
from utils.model_adapter import load_and_adapt_checkpoint

# 适配模型
checkpoint_path = "model_10000.pth"
output_path = "model_10000_adapted_60d.pth"

# 加载并适配
adapted_checkpoint = load_and_adapt_checkpoint(
    checkpoint_path,
    old_obs_dim=47,
    new_obs_dim=60,
    old_privileged_obs_dim=14,
    new_privileged_obs_dim=20,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 保存适配后的模型
torch.save(adapted_checkpoint, output_path)
print(f"适配后的模型已保存到: {output_path}") 