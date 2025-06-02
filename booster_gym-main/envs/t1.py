import os

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from .base_task import BaseTask

from utils.utils import apply_randomization


class T1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        
        # 验证设备设置
        print(f"[调试] T1初始化完成:")
        print(f"  device: {self.device}")
        print(f"  base_init_state device: {self.base_init_state.device}")
        print(f"  root_states device: {self.root_states.device}")
        print(f"  env_origins device: {self.env_origins.device}")

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        if self.cfg["env"].get("enable_soccer_field", False):
            soccer_field_options = gymapi.AssetOptions()
            soccer_field_options.fix_base_link = True
            soccer_field_options.disable_gravity = True
            soccer_field_asset = self.gym.load_asset(self.sim,asset_root, "soccer_field_half.urdf", soccer_field_options)
        else:
            soccer_field_asset = None
        if self.cfg["env"].get("enable_soccer_ball", False):
            soccer_ball_options = gymapi.AssetOptions()
            soccer_ball_options.angular_damping = 0.1
            soccer_ball_options.linear_damping = 0.1
            soccer_ball_asset = self.gym.load_asset(self.sim,asset_root, "soccer_ball.urdf", soccer_ball_options)
        else:
            soccer_ball_asset = None


        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])
        print(f"[调试] base_indice = {self.base_indice} (机器人'{asset_cfg['base_name']}'在机器人资产内的索引)")

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = [] # 存储球的handles
        self.field_handles = [] # 存储足球场的handles
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
                        # 创建足球场
            if soccer_field_asset is not None:
                field_pose = gymapi.Transform()
                field_pose.p = gymapi.Vec3(*pos)
                field_pose.p.z = -0.01  # 只调整高度，让场地表面贴近地面
                field_handle = self.gym.create_actor(env_handle, soccer_field_asset, field_pose, "soccer_field", i, 0, 1)
                self.field_handles.append(field_handle)
            else:
                self.field_handles.append(None)
            
            # 创建足球
            if soccer_ball_asset is not None:
                ball_pose = gymapi.Transform()  # 为每个环境创建新的球姿态
                ball_pose.p = gymapi.Vec3(*pos)  # 初始位置与环境原点相同
                
                ball_position_mode = self.cfg["env"].get("ball_position_mode")

                if ball_position_mode == "random" and self.cfg["env"].get("ball_position_range") is not None:
                    # 随机位置逻辑
                    pos_range = self.cfg["env"]["ball_position_range"]
                    rand_x = np.random.uniform(pos_range[0][0], pos_range[0][1])
                    rand_y = np.random.uniform(pos_range[1][0], pos_range[1][1])
                    rand_z = np.random.uniform(pos_range[2][0], pos_range[2][1])
                    
                    ball_pose.p.x += rand_x
                    ball_pose.p.y += rand_y
                    ball_pose.p.z = rand_z
                elif self.cfg["env"].get("ball_initial_position") is not None:
                    # 固定位置逻辑
                    ball_init_pos = self.cfg["env"]["ball_initial_position"]
                    ball_pose.p.x += ball_init_pos[0]
                    ball_pose.p.y += ball_init_pos[1]
                    ball_pose.p.z = ball_init_pos[2]
                else:
                    # 默认位置
                    ball_pose.p.x += 1.0
                    ball_pose.p.z = 0.11
                
                ball_handle = self.gym.create_actor(env_handle, soccer_ball_asset, ball_pose, "soccer_ball", i, 0, 2)
                self.ball_handles.append(ball_handle)
            else:
                ball_handle = None
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
   # 在_init_buffers中添加
        self.ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_local_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_local_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.has_ball = self.cfg["env"].get("enable_soccer_ball", False)
        if self.has_ball:# 足球相关缓冲区
            self.ball_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.ball_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
            self.ball_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.ball_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # 添加球门相关缓冲区
            # 根据soccer_field_half.urdf，只有一个球门位于y=9米处
            # 球门中心位置
            self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.goal_pos[:, 0] = 6.0    # x坐标（球门中心）
            self.goal_pos[:, 1] = 9.0    # y坐标（在边线上）
            self.goal_pos[:, 2] = 0.4    # z坐标（球门高度一半）
            
            # 球门宽度参数（用于判断进球）
            self.goal_width = 2.35  # 从URDF得出：7.175 - 4.825
            self.goal_height = 0.8  # 从URDF得出
            
            # 目标球门中心相对于机器人的方向向量
            self.goal_dir_relative = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # 机器人前进方向与球门方向的夹角
            self.ball_to_goal_angle = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
            # 球到目标球门中心的向量
            self.ball_to_goal_vec = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            
            # 如果有足球，确保观察维度正确
            # 注意：原本47维基础 + 可获取的观测(相对位置3维+相对速度3维+相对方向3维+夹角1维+球到目标3维) = 60维
            # 特权信息：原本14维 + 难以获取的观测(世界坐标球位置3维+球速度3维) = 20维
            if self.num_obs < 60:
                print(f"警告：观察空间维度可能不足，当前为{self.num_obs}，添加可获取的观测需要至少60维")
            if self.num_privileged_obs < 20: 
                print(f"警告：特权观察空间维度可能不足，当前为{self.num_privileged_obs}，添加特权信息需要至少20维")
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # 计算每个环境的actor数量
        num_actors_per_env = 1  # 机器人
        if self.cfg["env"].get("enable_soccer_field", False):
            num_actors_per_env += 1
        if self.cfg["env"].get("enable_soccer_ball", False):
            num_actors_per_env += 1
        self.num_actors_per_env = num_actors_per_env  # 保存为实例变量
        
        # 只提取机器人的root states（每个环境的第一个actor）
        robot_indices = torch.arange(0, self.num_envs * num_actors_per_env, num_actors_per_env, device=self.device)
        self.robot_indices = robot_indices  # 保存为成员变量
        self.robot_root_states = self.root_states[robot_indices]
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        
        # 使用-1让PyTorch自动计算每个环境的刚体总数
        body_states_all = gymtorch.wrap_tensor(body_state)
        self.total_num_bodies = body_states_all.shape[0] // self.num_envs
        self.body_states = body_states_all.view(self.num_envs, -1, 13)
        
        # 只提取机器人的刚体状态
        self.robot_body_states = self.body_states[:, :self.num_bodies, :]
        
        # 打印调试信息
        print(f"[调试] 每个环境的刚体总数: {self.total_num_bodies}")
        print(f"[调试] 机器人刚体数: {self.num_bodies}")
        print(f"[调试] 足球场和足球的刚体数: {self.total_num_bodies - self.num_bodies}")
        print(f"[调试] 每个环境的actor数: {num_actors_per_env}")
        print(f"[调试] root_states形状: {self.root_states.shape}")
        
        # 使用机器人的root states
        self.base_pos = self.robot_root_states[:, 0:3]
        self.base_quat = self.robot_root_states[:, 3:7]
        self.feet_pos = self.robot_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.robot_body_states[:, self.feet_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.robot_root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        # 使用total_num_bodies而不是self.num_bodies，确保与实际的刚体数量匹配
        self.pushing_forces = torch.zeros(self.num_envs, self.total_num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.total_num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def reset(self):
        """Reset all robots"""
        print(f"[调试] reset: self.num_envs={self.num_envs}, self.device={self.device}")
        if self.device != "cpu":
            torch.cuda.synchronize()
        env_ids = torch.arange(self.num_envs, device=self.device)
        print(f"[调试] reset: env_ids创建完成, shape={env_ids.shape}, device={env_ids.device}")
        self._reset_idx(env_ids)
        if self.device != "cpu":
            torch.cuda.synchronize()
        self._resample_commands()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.robot_root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        # 确保num_actors_per_env已初始化
        if not hasattr(self, 'num_actors_per_env'):
            raise RuntimeError("num_actors_per_env not initialized")
            
        # 确保env_ids不为空
        if len(env_ids) == 0:
            return
        
        # 添加调试信息
        print(f"[调试] _reset_root_states: env_ids.shape={env_ids.shape}, device={env_ids.device}")
        print(f"[调试] _reset_root_states: num_actors_per_env={self.num_actors_per_env}")
        print(f"[调试] _reset_root_states: self.device={self.device}")
        
        # 确保env_ids在正确的设备上
        if env_ids.device != self.device:
            env_ids = env_ids.to(self.device)
            
        # 使用向量化操作获取机器人在root_states中的索引
        robot_indices = (env_ids * self.num_actors_per_env).long()
        
        # 边界检查
        max_idx = robot_indices.max().item() if len(robot_indices) > 0 else -1
        if max_idx >= self.root_states.shape[0]:
            raise RuntimeError(f"robot_indices越界: max_idx={max_idx}, root_states.shape={self.root_states.shape}")
        
        # 更新root_states中机器人的部分
        self.root_states[robot_indices] = self.base_init_state
        self.root_states[robot_indices, :2] += self.env_origins[env_ids, :2]
        self.root_states[robot_indices, :2] = apply_randomization(self.root_states[robot_indices, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[robot_indices, 2] += self.terrain.terrain_heights(self.root_states[robot_indices, :2])
        self.root_states[robot_indices, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
        self.root_states[robot_indices, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        # 同步更新robot_root_states
        self.robot_root_states[env_ids] = self.root_states[robot_indices]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
            
        out_x_min = self.robot_root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.robot_root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.robot_root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.robot_root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        
        # 更新robot_root_states
        self.robot_root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.robot_root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.robot_root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.robot_root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        
        # 同步更新root_states中对应的机器人状态
        # 找出所有需要更新的环境
        need_update = out_x_min | out_x_max | out_y_min | out_y_max
        if need_update.any():
            # 使用向量化操作更新root_states
            env_ids = torch.arange(self.num_envs, device=self.device)[need_update]
            robot_indices = (env_ids * self.num_actors_per_env).long()
            self.root_states[robot_indices] = self.robot_root_states[env_ids]
        
        # 更新body_states
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        if self.cfg["commands"]["curriculum"]:
            self._resample_curriculum_commands(env_ids)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                x = self.env_curriculum_level[env_ids[i], 0] + self.cfg["commands"]["lin_vel_levels"]
                y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _resample_curriculum_commands(self, env_ids):
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["ang_vel_resolution"]

    def step(self, actions):
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 更新robot_root_states（因为root_states已经被刷新）
        robot_indices = torch.arange(0, self.num_envs * self.num_actors_per_env, self.num_actors_per_env, device=self.device)
        self.robot_root_states[:] = self.root_states[robot_indices]
        
        # 更新robot_body_states（因为body_states已经被刷新）
        self.robot_body_states[:] = self.body_states[:, :self.num_bodies, :]
        
        self.base_pos[:] = self.robot_root_states[:, 0:3]
        self.base_quat[:] = self.robot_root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._resample_commands()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.robot_root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            # 更新robot_root_states
            self.robot_root_states[:, 7:10] = apply_randomization(self.robot_root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.robot_root_states[:, 10:13] = apply_randomization(self.robot_root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            
            # 同步更新root_states中的机器人状态
            robot_indices = torch.arange(0, self.num_envs * self.num_actors_per_env, self.num_actors_per_env, device=self.device)
            self.root_states[robot_indices] = self.robot_root_states
            
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.robot_body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.robot_body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.robot_root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        """Computes observations"""
        commands_scale = torch.tensor(
            [self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["ang_vel"]],
            device=self.device,
        )
        
        # 基础观察（47维）
        base_obs = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.commands[:, :3] * commands_scale,
                (torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )
        
        # 如果启用了足球，添加足球相关观察
        if self.has_ball:
            # 更新足球状态
            # root_states包含所有环境的所有actor
            # 每个环境的actor顺序：机器人[0], 足球场[1](如果有), 足球[1或2]
            if self.cfg["env"].get("enable_soccer_field", False):
                # 有足球场：机器人(0) + 足球场(1) + 足球(2) = 每环境3个actor
                actors_per_env = 3
                ball_idx_in_env = 2
            else:
                # 无足球场：机器人(0) + 足球(1) = 每环境2个actor
                actors_per_env = 2
                ball_idx_in_env = 1
            
            # 获取每个环境中球的索引
            ball_indices = torch.arange(self.num_envs, device=self.device) * actors_per_env + ball_idx_in_env
            
            # 获取球的状态
            self.ball_pos[:] = self.root_states[ball_indices, 0:3]
            self.ball_vel[:] = self.root_states[ball_indices, 7:10]
            
            # 计算球在机器人局部坐标系中的位置
            ball_relative_pos = self.ball_pos - self.base_pos
            self.ball_local_pos[:] = quat_rotate_inverse(self.base_quat, ball_relative_pos)
            
            # 计算球相对于机器人的速度（在局部坐标系中）
            ball_relative_vel = self.ball_vel - self.robot_root_states[:, 7:10]
            self.ball_local_vel = quat_rotate_inverse(self.base_quat, ball_relative_vel)
            
            # 计算球门方向相关的观察
            # 1. 球门中心相对于机器人躯干的3D方向
            goal_relative_pos = self.goal_pos - self.base_pos
            self.goal_dir_relative[:] = quat_rotate_inverse(self.base_quat, goal_relative_pos)
            # 归一化为单位向量
            goal_dist = torch.norm(self.goal_dir_relative, dim=1, keepdim=True)
            self.goal_dir_relative = self.goal_dir_relative / (goal_dist + 1e-6)
            
            # 2. 机器人前进方向与球门方向的夹角
            # 机器人局部坐标系的前进方向是x轴正方向
            forward_dir_local = torch.zeros_like(self.base_pos)
            forward_dir_local[:, 0] = 1.0
            # 转换到世界坐标系
            forward_dir_world = quat_rotate(self.base_quat, forward_dir_local)
            # 计算世界坐标系中的球门方向
            goal_dir_world = self.goal_pos - self.base_pos
            goal_dir_world_norm = goal_dir_world / (torch.norm(goal_dir_world, dim=1, keepdim=True) + 1e-6)
            # 计算夹角余弦值
            cos_angle = torch.sum(forward_dir_world * goal_dir_world_norm, dim=1, keepdim=True)
            # 转换为弧度角
            self.ball_to_goal_angle[:] = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
            
            # 3. 球到球门中心的向量
            self.ball_to_goal_vec[:] = self.goal_pos - self.ball_pos
            
            # 添加足球观察（13维）
            ball_obs = torch.cat(
                (
                    self.ball_local_pos * 0.5,      # 球相对位置 (3维)，缩放
                    self.ball_local_vel * 0.2,      # 球相对速度 (3维)，缩放
                    self.goal_dir_relative,         # 球门方向单位向量 (3维)
                    self.ball_to_goal_angle,        # 机器人朝向与球门夹角 (1维)
                    self.ball_to_goal_vec * 0.1,   # 球到球门向量 (3维)，缩放
                ),
                dim=-1,
            )
            
            self.obs_buf = torch.cat((base_obs, ball_obs), dim=-1)
        else:
            self.obs_buf = base_obs
        
        # 特权观察保持不变
        if self.has_ball:
            # 当有足球时，特权观察包含额外的球信息（20维 = 14基础 + 6足球）
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_mass_scaled,                       # 4维
                    apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],  # 3维
                    apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),  # 1维
                    self.pushing_forces[:, self.base_indice, :] * self.cfg["normalization"]["push_force"],   # 3维
                    self.pushing_torques[:, self.base_indice, :] * self.cfg["normalization"]["push_torque"], # 3维
                    self.ball_pos * 0.5,  # 球世界坐标位置 (3维)，缩放
                    self.ball_vel * 0.5,  # 球世界坐标速度 (3维)，缩放
                ),
                dim=-1,
            )
        else:
            # 无足球时，保持原来的14维特权观察
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_mass_scaled,
                    apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                    apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                    self.pushing_forces[:, self.base_indice, :] * self.cfg["normalization"]["push_force"],
                    self.pushing_torques[:, self.base_indice, :] * self.cfg["normalization"]["push_torque"],
                ),
                dim=-1,
            )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(-torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(-torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.robot_root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        return torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1)

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()
    def _reward_approach_ball(self):
    # """靠近球的奖励 - 鼓励机器人接近足球  
    # 在Phase-2中启用，权重较高"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # 计算机器人到球的距离（使用局部坐标系中的位置）
        dist_to_ball = torch.norm(self.ball_local_pos, dim=1)
        # 使用高斯函数将距离转换为奖励，距离越近奖励越高
        approach_sigma = self.cfg["rewards"].get("approach_sigma", 0.5)  # 可在yaml中配置
        # 最大奖励为1.0，随距离增加呈指数衰减
        return torch.exp(-torch.square(dist_to_ball) / approach_sigma)
        
    def _reward_face_ball(self):
        """面向球的奖励 - 鼓励机器人正面朝向球
        在Phase-2中启用，与approach_ball配合使用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 机器人前向方向在局部坐标系中是(1,0,0)
        # ball_local_pos已经是球在机器人局部坐标系中的位置
        forward_vec = torch.zeros_like(self.ball_local_pos)
        forward_vec[:, 0] = 1.0  # x轴正方向是机器人的前向
        
        # 计算球的方向向量（需要先归一化）
        ball_dir = self.ball_local_pos.clone()
        ball_dist = torch.norm(ball_dir, dim=1, keepdim=True) + 1e-6
        ball_dir = ball_dir / ball_dist
        # 计算前向向量与球方向向量的点积(余弦值)
        # 完全朝向球时为1，垂直时为0，背对时为-1
        cos_angle = torch.sum(forward_vec * ball_dir, dim=1)
        
        # 将余弦值裁剪到[0,1]范围，只奖励正面朝向球
        return torch.clamp(cos_angle, min=0.0)
        
    def _reward_align_goal(self):
        """球门对准奖励 - 鼓励机器人让球、自己和球门在一条直线上
        在Phase-3中启用，为踢球做准备"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 计算两个关键向量
        # 1. 机器人到球的方向（机器人应该面对的方向）
        robot_to_ball_dir = self.ball_local_pos.clone()
        robot_to_ball_dist = torch.norm(robot_to_ball_dir, dim=1, keepdim=True) + 1e-6
        robot_to_ball_dir = robot_to_ball_dir / robot_to_ball_dist
        
        # 2. 球到球门的方向（应该踢球的方向）
        # 我们已经在观测空间中有ball_to_goal_vec
        ball_to_goal_dir = self.ball_to_goal_vec.clone()
        ball_to_goal_dist = torch.norm(ball_to_goal_dir, dim=1, keepdim=True) + 1e-6
        ball_to_goal_dir = ball_to_goal_dir / ball_to_goal_dist
        
        # 计算这两个向量的夹角余弦值
        # 球、机器人、球门完全对齐时，余弦值为1
        cos_angle = torch.sum(robot_to_ball_dir * ball_to_goal_dir, dim=1)
        
        # 我们希望机器人站在球后面，朝向球门
        # 只有当角度小于90度时（余弦值>0）才给予奖励
        return torch.clamp(cos_angle, min=0.0)
        
    def _reward_kick_velocity(self):
        """踢球速度奖励 - 鼓励球朝向球门方向移动
        在Phase-3和Phase-4中启用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 计算球的速度在球门方向上的投影
        ball_vel_world = self.ball_vel
        
        # 球到球门的方向（单位向量）
        ball_to_goal_dir = self.ball_to_goal_vec.clone()
        ball_to_goal_dist = torch.norm(ball_to_goal_dir, dim=1, keepdim=True) + 1e-6
        ball_to_goal_dir = ball_to_goal_dir / ball_to_goal_dist
        
        # 计算球速在球门方向上的投影分量
        vel_proj = torch.sum(ball_vel_world * ball_to_goal_dir, dim=1)
        
        # 只有当球朝向球门移动时才给予奖励（速度投影为正）
        # 且奖励与速度成正比，但设置上限
        max_velocity = 5.0  # 可在yaml中配置
        return torch.clamp(vel_proj, min=0.0, max=max_velocity) / max_velocity
        
    def _reward_goal_scored(self):
        """进球奖励 - 当球进入球门时给予高额奖励
        在Phase-4中启用，是最终的训练目标"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 球门位置参数（从URDF获得）
        goal_x = 6.0  # 球门中心x坐标
        goal_y = 9.0  # 球门在y=9米的边线上
        goal_width_half = self.goal_width / 2  # 1.175米
        goal_height = self.goal_height  # 0.8米
        
        # 检查球是否在球门范围内
        # x方向：球门宽度范围内
        in_x_range = torch.abs(self.ball_pos[:, 0] - goal_x) < goal_width_half + 0.1  # 稍微宽松一点
        # y方向：球要越过球门线
        in_y_range = self.ball_pos[:, 1] > goal_y - 0.2  # 球门线前0.2米内
        # z方向：球高度在球门高度内
        in_z_range = (self.ball_pos[:, 2] > 0.0) & (self.ball_pos[:, 2] < goal_height + 0.1)
        
        # 判断球是否进球
        scored = in_x_range & in_y_range & in_z_range
        
        # 进球给予固定奖励1.0，未进球为0
        # 注意：实际训练时，可以在T1.yaml中设置很大的系数(如30)
        return scored.float()
        
    def _reward_dribbling(self):
        """带球奖励 - 鼓励机器人在控制球的同时移动
        可选奖励，在Phase-4中启用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 定义控制范围 - 球在机器人前方适当距离内
        control_dist_min = 0.3
        control_dist_max = 0.8
        
        # 计算球到机器人的距离
        ball_dist = torch.norm(self.ball_local_pos, dim=1)
        
        # 球在控制范围内的mask
        in_control = (ball_dist > control_dist_min) & (ball_dist < control_dist_max)
        
        # 机器人和球都在移动的情况(使用x方向速度作为指标)
        robot_moving = torch.abs(self.base_lin_vel[:, 0]) > 0.3  # 机器人在移动
        ball_moving = torch.abs(self.ball_vel[:, 0]) > 0.2  # 球在移动
        
        # 带球移动的奖励
        dribbling = in_control & robot_moving & ball_moving
        
        return dribbling.float()
        
    def _reward_ball_position_z(self):
        """球高度惩罚 - 惩罚球过高，鼓励低平射门
        辅助奖励，根据需要启用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 理想高度是球半径
        ball_radius = 0.11  # 从soccer_ball.urdf获得
        ideal_height = ball_radius
        
        # 计算球高度与理想高度的差距，并惩罚
        height_diff = torch.abs(self.ball_pos[:, 2] - ideal_height)
        
        # 转换为[0,1]范围的惩罚，差距越大惩罚越大
        height_sigma = 0.5  # 可在yaml中配置
        # 注意：这里返回的是惩罚，T1.yaml中应设为负权重
        return torch.exp(-torch.square(height_diff) / height_sigma)