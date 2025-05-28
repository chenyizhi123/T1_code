# 标准半场足球场 URDF 文件说明

## 文件概述
`soccer_field_standard_half.urdf` 是一个符合FIFA标准的半场足球场URDF文件，专为机器人足球仿真设计。

## 主要特点

### 1. 标准尺寸（缩放比例 1:10）
- **半场尺寸**: 52.5m × 68m → 5.25m × 6.8m
- **球门宽度**: 7.32m → 0.732m
- **球门高度**: 2.44m → 0.244m
- **大禁区**: 16.5m × 40.3m → 1.65m × 4.03m
- **小禁区**: 5.5m × 18.32m → 0.55m × 1.832m
- **点球点距离**: 11m → 1.1m
- **中圈半径**: 9.15m → 0.915m

### 2. 完整的场地标线
- 边线和底线（球门线）
- 中线
- 大禁区（罚球区）线
- 小禁区（球门区）线
- 点球点
- 中心点

### 3. 标准球门结构
- 左右立柱（圆柱形）
- 横梁（圆柱形）
- 球门网（简化的半透明箱体）

### 4. 场地围栏
- 左右边界围栏
- 球门后方围栏
- 防止球出界

## 使用方法

### 1. 在ROS中使用
```xml
<launch>
  <param name="robot_description" 
         textfile="$(find your_package)/resources/T1/soccer_field_standard_half.urdf" />
  
  <node name="spawn_field" pkg="gazebo_ros" type="spawn_model"
        args="-urdf -model soccer_field -param robot_description" />
</launch>
```

### 2. 在Python中加载
```python
import pybullet as p
import pybullet_data

# 连接物理引擎
physicsClient = p.connect(p.GUI)

# 加载足球场
field_id = p.loadURDF("soccer_field_standard_half.urdf", 
                      basePosition=[0, 0, 0],
                      useFixedBase=True)
```

### 3. 坐标系说明
- 原点位于半场中心（即中线与边线中点的交点）
- X轴：指向对方球门为负方向
- Y轴：从左边线到右边线为正方向
- Z轴：垂直向上为正方向

## 材质颜色
- **草地**: 深绿色 (0.133, 0.545, 0.133)
- **标线**: 白色 (1.0, 1.0, 1.0)
- **球门**: 白色 (1.0, 1.0, 1.0)
- **球网**: 半透明白色 (1.0, 1.0, 1.0, 0.3)
- **围栏**: 灰色 (0.5, 0.5, 0.5, 0.8)

## 注意事项
1. 本文件不依赖外部mesh文件，所有几何形状都使用基本图元
2. 缩放比例为1:10，便于室内仿真环境使用
3. 所有碰撞体都已正确设置，可用于物理仿真
4. 球门网使用简化的箱体表示，主要起视觉效果作用

## 自定义修改
如需修改场地尺寸，请注意保持各部分的比例关系，确保符合足球场的标准规范。

## 相关文件
- `soccer_ball.urdf`: 配套的足球URDF文件
- `T1_locomotion.urdf`: 机器人模型文件 