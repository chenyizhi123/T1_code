#!/usr/bin/env python3
"""
测试T1足球环境中各个物体的自由度(DOF)数量
"""

import os
import sys
sys.path.append('.')

from isaacgym import gymapi, gymtorch
import torch
import yaml

def test_dof_counts():
    """测试各个资产的DOF数量"""
    
    print("=" * 60)
    print("T1足球环境 DOF数量测试")
    print("=" * 60)
    
    # 初始化Isaac Gym
    gym = gymapi.acquire_gym()
    
    # 创建仿真
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.002
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.use_gpu_pipeline = True
    
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    # 资产路径
    asset_root = "resources/T1"
    
    # 1. 测试机器人DOF
    print("\n1. 机器人资产分析:")
    print("-" * 30)
    
    robot_options = gymapi.AssetOptions()
    robot_options.default_dof_drive_mode = 3
    robot_options.collapse_fixed_joints = True
    robot_options.fix_base_link = False
    robot_options.disable_gravity = False
    
    robot_asset = gym.load_asset(sim, asset_root, "T1_locomotion.urdf", robot_options)
    robot_dofs = gym.get_asset_dof_count(robot_asset)
    robot_bodies = gym.get_asset_rigid_body_count(robot_asset)
    robot_dof_names = gym.get_asset_dof_names(robot_asset)
    
    print(f"  机器人DOF数量: {robot_dofs}")
    print(f"  机器人刚体数量: {robot_bodies}")
    print(f"  机器人DOF名称: {robot_dof_names}")
    
    # 2. 测试足球场DOF  
    print("\n2. 足球场资产分析:")
    print("-" * 30)
    
    field_options = gymapi.AssetOptions()
    field_options.fix_base_link = True
    field_options.disable_gravity = True
    
    try:
        field_asset = gym.load_asset(sim, asset_root, "soccer_field_half.urdf", field_options)
        field_dofs = gym.get_asset_dof_count(field_asset)
        field_bodies = gym.get_asset_rigid_body_count(field_asset)
        field_dof_names = gym.get_asset_dof_names(field_asset)
        
        print(f"  足球场DOF数量: {field_dofs}")
        print(f"  足球场刚体数量: {field_bodies}")
        print(f"  足球场DOF名称: {field_dof_names}")
    except Exception as e:
        print(f"  足球场加载失败: {e}")
        field_dofs = 0
        field_bodies = 0
    
    # 3. 测试足球DOF
    print("\n3. 足球资产分析:")
    print("-" * 30)
    
    ball_options = gymapi.AssetOptions()
    ball_options.angular_damping = 0.1
    ball_options.linear_damping = 0.1
    
    try:
        ball_asset = gym.load_asset(sim, asset_root, "soccer_ball.urdf", ball_options)
        ball_dofs = gym.get_asset_dof_count(ball_asset)
        ball_bodies = gym.get_asset_rigid_body_count(ball_asset)
        ball_dof_names = gym.get_asset_dof_names(ball_asset)
        
        print(f"  足球DOF数量: {ball_dofs}")
        print(f"  足球刚体数量: {ball_bodies}")
        print(f"  足球DOF名称: {ball_dof_names}")
    except Exception as e:
        print(f"  足球加载失败: {e}")
        ball_dofs = 0
        ball_bodies = 0
    
    # 4. 创建测试环境来验证实际DOF数量
    print("\n4. 环境实际DOF测试:")
    print("-" * 30)
    
    # 创建一个简单的环境
    env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
    env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    # 添加机器人
    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(0, 0, 0.72)
    robot_handle = gym.create_actor(env, robot_asset, robot_pose, "robot", 0, 0, 0)
    
    # 添加足球场
    if 'field_asset' in locals():
        field_pose = gymapi.Transform()
        field_pose.p = gymapi.Vec3(0, 0, 0)
        field_handle = gym.create_actor(env, field_asset, field_pose, "field", 0, 0, 1)
    
    # 添加足球
    if 'ball_asset' in locals():
        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(1, 0, 0.2)
        ball_handle = gym.create_actor(env, ball_asset, ball_pose, "ball", 0, 0, 2)
    
    # 准备仿真
    gym.prepare_sim(sim)
    
    # 获取DOF状态张量
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(dof_state_tensor)
    
    print(f"  环境总DOF数量: {dof_states.shape[0]}")
    print(f"  DOF状态张量形状: {dof_states.shape}")
    
    # 获取root state张量
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_tensor)
    
    print(f"  环境总Actor数量: {root_states.shape[0]}")
    print(f"  Root状态张量形状: {root_states.shape}")
    
    # 获取body state张量
    body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    body_states = gymtorch.wrap_tensor(body_state_tensor)
    
    print(f"  环境总刚体数量: {body_states.shape[0]}")
    print(f"  Body状态张量形状: {body_states.shape}")
    
    # 5. 总结
    print("\n5. 总结:")
    print("-" * 30)
    total_expected_dofs = robot_dofs + field_dofs + ball_dofs
    print(f"  预期总DOF: {robot_dofs}(机器人) + {field_dofs}(足球场) + {ball_dofs}(足球) = {total_expected_dofs}")
    print(f"  实际总DOF: {dof_states.shape[0]}")
    
    total_expected_bodies = robot_bodies + field_bodies + ball_bodies  
    print(f"  预期总刚体: {robot_bodies}(机器人) + {field_bodies}(足球场) + {ball_bodies}(足球) = {total_expected_bodies}")
    print(f"  实际总刚体: {body_states.shape[0]}")
    
    print(f"  总Actor数: {root_states.shape[0]}")
    
    if total_expected_dofs == dof_states.shape[0]:
        print("  ✅ DOF数量匹配!")
    else:
        print("  ❌ DOF数量不匹配!")
        
    if total_expected_bodies == body_states.shape[0]:
        print("  ✅ 刚体数量匹配!")
    else:
        print("  ❌ 刚体数量不匹配!")
    
    # 清理
    gym.destroy_sim(sim)
    print("\n测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_dof_counts() 