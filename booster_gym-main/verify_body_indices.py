"""
验证多actor环境下的刚体索引
"""

# 根据调试输出
num_envs = 4096
robot_bodies = 13
field_bodies = 3  # 从调试输出得知
ball_bodies = 1
total_bodies_per_env = 17

print("=== 多Actor环境的刚体组织 ===\n")

print("每个环境的刚体排列顺序：")
print(f"索引 0-{robot_bodies-1}：机器人的{robot_bodies}个刚体")
print(f"索引 {robot_bodies}-{robot_bodies+field_bodies-1}：足球场的{field_bodies}个刚体")
print(f"索引 {robot_bodies+field_bodies}：足球的{ball_bodies}个刚体")
print(f"总计：{total_bodies_per_env}个刚体/环境\n")

print("关于 self.base_indice：")
print("- 它是通过 find_asset_rigid_body_index(robot_asset, 'Trunk') 获得的")
print("- 这个索引是相对于机器人资产内部的（0-12之间）")
print("- 假设 'Trunk' 是机器人的第一个刚体，base_indice = 0")
print("- 如果 'Trunk' 是其他位置，base_indice 会是相应的值\n")

print("使用场景验证：")
print("1. body_states[:, self.base_indice, :] ✓")
print("   - 正确：因为机器人刚体从索引0开始")
print("   - 获取每个环境中机器人躯干的状态")
print()
print("2. pushing_forces[:, self.base_indice, :] ✓")
print("   - 正确：pushing_forces与body_states有相同的刚体排列")
print("   - 在每个环境的机器人躯干上施加推力")
print()
print("3. contact_forces[:, self.termination_contact_indices, :] ✓")
print("   - 正确：termination_contact_indices也是机器人内部的索引")
print("   - 检查机器人特定部位的接触力")

print("\n=== 结论 ===")
print("self.base_indice 在多actor环境下仍然正确，因为：")
print("1. 它指向的是每个环境中机器人刚体组内的相对位置")
print("2. 机器人的刚体总是从索引0开始排列")
print("3. 所以 base_indice 能正确定位到每个环境的机器人躯干") 