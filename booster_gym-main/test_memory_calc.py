"""
计算PhysX所需的碰撞对内存
"""

# 环境参数
num_envs = 4096
robot_bodies = 13
# 从soccer_field_half.urdf分析：
# - field_base: 1个刚体
# - field_lines: 通过fixed关节连接，不增加刚体
# - goal_structure: 通过fixed关节连接，不增加刚体
field_bodies = 1  # 实际只有1个刚体！
ball_bodies = 1
total_bodies_per_env = robot_bodies + field_bodies + ball_bodies

print(f"环境数量: {num_envs}")
print(f"每个环境的刚体数:")
print(f"  - 机器人: {robot_bodies}")
print(f"  - 足球场: {field_bodies}")
print(f"  - 足球: {ball_bodies}")
print(f"  - 总计: {total_bodies_per_env}")
print(f"总刚体数: {num_envs * total_bodies_per_env}")

# 碰撞对计算
# 理论最大值：n*(n-1)/2，但实际上不是所有物体都会碰撞
# PhysX使用空间分区来减少检测数量

# 保守估计：每个物体平均与10个其他物体进行碰撞检测
avg_collision_checks_per_body = 10
estimated_pairs = (num_envs * total_bodies_per_env * avg_collision_checks_per_body) // 2

print(f"\n基于刚体数的估计碰撞对: {estimated_pairs:,}")
print(f"PhysX实际要求的容量: 35,889,149")
print(f"差异: {35889149 - estimated_pairs:,}")

# 这个巨大的差异可能是因为：
print("\n可能的原因：")
print("1. PhysX内部可能为每个环境创建额外的碰撞检测结构")
print("2. 机器人的13个刚体之间也会相互检测（自碰撞）")
print("3. PhysX可能预分配更多空间以避免动态扩展")

# 内存估算（每个碰撞对约32字节）
memory_mb = (35889149 * 32) / (1024 * 1024)
print(f"\nPhysX要求的内存: {memory_mb:.1f} MB")
print(f"建议：必须将foundLostAggregatePairsCapacity设置为至少35889149") 