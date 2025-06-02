import os
# 设置CUDA调试模式
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('./booster_gym-main')

import torch
import yaml
from envs.t1 import T1

# 加载配置
with open('booster_gym-main/configs/T1.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# 更新为测试配置
cfg['env']['num_envs'] = 1  # 只用1个环境进行测试
cfg['basic']['headless'] = True
cfg['viewer']['record_video'] = False

print(f"创建环境，num_envs: {cfg['env']['num_envs']}")
print(f"device: {cfg['basic']['sim_device']}")

try:
    # 创建环境
    env = T1(cfg)
    print("环境创建成功!")
    
    print(f"num_envs: {env.num_envs}")
    print(f"num_actors_per_env: {env.num_actors_per_env}")
    print(f"device: {env.device}")
    
    # 测试reset
    print("\n测试reset...")
    obs, extras = env.reset()
    print("Reset成功!")
    
    print(f"obs shape: {obs.shape}")
    print(f"extras keys: {list(extras.keys())}")
    
    # 测试step
    print("\n测试step...")
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    obs, rew, reset, extras = env.step(actions)
    print("Step成功!")
    
except Exception as e:
    import traceback
    print(f"\n错误: {e}")
    print("\n完整错误信息:")
    traceback.print_exc() 