import torch
from utils.model_adapter import ModelAdapter

# 加载原始模型
checkpoint_path = "model_10000.pth"
output_path = "model_10000_adapted_60d_fixed.pth"

print(f"加载模型: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 打印checkpoint的结构
print("\n检查checkpoint结构:")
print(f"Keys: {checkpoint.keys()}")

# 检查是否有嵌套的'model'键
if 'model' in checkpoint:
    print("\n发现'model'键，提取state_dict")
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# 创建一个新的checkpoint，只包含state_dict
new_checkpoint = {'model': state_dict}

# 检查actor和critic的输入维度
if 'actor.0.weight' in state_dict:
    print(f"\nActor输入维度: {state_dict['actor.0.weight'].shape[1]}")
if 'critic.0.weight' in state_dict:
    print(f"Critic输入维度: {state_dict['critic.0.weight'].shape[1]}")

# 手动适配权重
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 适配actor
old_obs_dim = 47
new_obs_dim = 60
if 'actor.0.weight' in state_dict and state_dict['actor.0.weight'].shape[1] == old_obs_dim:
    old_weight = state_dict['actor.0.weight']
    new_weight = torch.zeros(old_weight.shape[0], new_obs_dim, device=device)
    new_weight[:, :old_obs_dim] = old_weight.to(device)
    new_weight[:, old_obs_dim:] = torch.randn(old_weight.shape[0], new_obs_dim - old_obs_dim, device=device) * 0.01
    state_dict['actor.0.weight'] = new_weight
    print(f"\n适配Actor: {old_weight.shape} -> {new_weight.shape}")

# 适配critic
old_privileged_obs_dim = 14
new_privileged_obs_dim = 20
old_total_dim = old_obs_dim + old_privileged_obs_dim  # 61
new_total_dim = new_obs_dim + new_privileged_obs_dim  # 80

if 'critic.0.weight' in state_dict and state_dict['critic.0.weight'].shape[1] == old_total_dim:
    old_weight = state_dict['critic.0.weight']
    new_weight = torch.zeros(old_weight.shape[0], new_total_dim, device=device)
    
    # 复制观察部分
    new_weight[:, :old_obs_dim] = old_weight[:, :old_obs_dim].to(device)
    # 新增观察维度
    new_weight[:, old_obs_dim:new_obs_dim] = torch.randn(old_weight.shape[0], new_obs_dim - old_obs_dim, device=device) * 0.01
    # 复制特权观察部分
    new_weight[:, new_obs_dim:new_obs_dim+old_privileged_obs_dim] = old_weight[:, old_obs_dim:old_obs_dim+old_privileged_obs_dim].to(device)
    # 新增特权观察维度
    new_weight[:, new_obs_dim+old_privileged_obs_dim:] = torch.randn(old_weight.shape[0], new_privileged_obs_dim - old_privileged_obs_dim, device=device) * 0.01
    
    state_dict['critic.0.weight'] = new_weight
    print(f"适配Critic: {old_weight.shape} -> {new_weight.shape}")

# 将state_dict移回CPU再保存
for key in state_dict:
    if isinstance(state_dict[key], torch.Tensor):
        state_dict[key] = state_dict[key].cpu()

# 保存适配后的模型
torch.save({'model': state_dict}, output_path)
print(f"\n适配后的模型已保存到: {output_path}")
print("现在可以使用: python train.py --task=T1_phase1 --checkpoint=model_10000_adapted_60d_fixed.pth") 