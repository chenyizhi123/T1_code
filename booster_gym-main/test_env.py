import torch
import yaml
from envs.t1 import T1

# 加载配置
with open("envs/T1_phase1.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# 减少环境数量以便测试
cfg["env"]["num_envs"] = 4
cfg["basic"]["headless"] = True

print("创建环境...")
try:
    env = T1(cfg)
    print(f"环境创建成功!")
    print(f"num_envs: {env.num_envs}")
    print(f"num_actors_per_env: {env.num_actors_per_env}")
    print(f"device: {env.device}")
    
    print("\n测试reset...")
    obs, extras = env.reset()
    print(f"观察空间形状: {obs.shape}")
    print(f"期望形状: ({env.num_envs}, {env.num_obs})")
    
    print("\n测试step...")
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    obs, rewards, dones, extras = env.step(actions)
    print(f"step完成!")
    print(f"rewards形状: {rewards.shape}")
    print(f"dones形状: {dones.shape}")
    
    print("\n环境测试通过!")
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc() 