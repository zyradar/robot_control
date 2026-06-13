"""机械臂关节限制量"""
Joint_Limit = {
  "limit_rad":
  {
    # "joint_name":[min_rad_position, max_rad_position]
    # "关节名":[最小弧度位置, 最大弧度位置]
    "joint_1" :
      [-3.106, 3.106],
    "joint_2" :
      [-2.2689, 2.2689],
    "joint_3" :
      [-2.356, 2.356],
    "joint_4" :
      [-3.106, 3.106],
    "joint_5" :
      [-2.234, 2.234],
    "joint_6" :
      [-6.28, 6.28]
  },
  "limit_rangle":
  {
    # "joint_name":[min_rangle_position, max_rangle_position]
    # "关节名":[最小角度位置, 最大角度位置]
    "joint_1" :
      [-178.0, 178.0],
    "joint_2":
      [-130.0, 130.0],
    "joint_3" :
      [-135.0, 140.0],
    "joint_4" :
      [-178.0, 178.0],
    "joint_5" :
      [-128.0, 128.0],
    "joint_6" :
      [-360.0, 360.0]
  },
  "limit_speed":
  {
    # "joint_name":[max_rad_acceleration, max_rangle_acceleration]
    # "关节名":[最大弧度速度, 最大角度速度]
    "joint_1" :
      [3.14, 600.0],
    "joint_2" :
      [3.14, 600.0],
    "joint_3" :
      [3.92, 600.0],
    "joint_4" :
      [3.92, 600.0],
    "joint_5" :
      [3.92, 600.0],
    "joint_6" :
      [3.92, 600.0]
  },
}

Joint_init_position = {
  # 机械臂出厂位置
  "arm_factory_init":{
  "joint_1" : 0.0,
  "joint_2" : -2.18126,
  "joint_3" : 2.26851,
  "joint_4" : 0.0,
  "joint_5" : 0.7852,
  "joint_6" : 0.0, 
  "position":[-0.5, 0.0, 0.5], 
  "orientation": [0.926156, 0.0, -0.37714, -3.0], 
  },

  # 机械臂全0位置
  "arm_zero_init":{
  "joint_1" : 0.0,
  "joint_2" : 0.0,
  "joint_3" : 0.0,
  "joint_4" : 0.0,
  "joint_5" : 0.0,
  "joint_6" : 0.0, 
  "position": [0.0, 0.0, 1.06],     # x,y,z
  "orientation": [0.0, 0.0, 1.0, 0.0], 
  },

  # 机械臂初始位置
  "arm_user_init":{
  "joint_1" : 0.0,
  "joint_2" : -0.6,
  "joint_3" : 1.8432,
  "joint_4" : 0.0,
  "joint_5" : 1.125,
  "joint_6" : 0.0, 
  "position": [-0.3, 0.0, 0.266],     # x,y,z
  "orientation": [0.926156, 0.0, -0.37714, -3.0], 
  },

  # 机械臂工作位置
  "arm_action_init":{
  "joint_1" : -1.6,
  "joint_2" : -0.2,
  "joint_3" : 1.8432,
  "joint_4" : 0.0,
  "joint_5" : 1.125,
  "joint_6" : 0.0, 
  "position": [0.0084, 0.2876, 0.14466],     # x,y,z
  "orientation": [0.926156, 0.0, -0.37714, 0.0], 
  },

  # 玩偶正向位置tool1
  "arm_tool_1":{
  "joint_1" : -1.62,
  "joint_2" : 1.88,
  "joint_3" : 0.763,
  "joint_4" : -0.01265,
  "joint_5" : -0.0013,
  "joint_6" : 0.0, 
  "position": [0.026, 0.511, -0.3345],     # x,y,z
  "orientation": [-0.678, 0.6928, 0.16642, 0.1804], 
  },

  # 玩偶斜向位置tool2
  "arm_tool_2":{
  "joint_1" : -1.1826,
  "joint_2" : 1.9335,
  "joint_3" : 0.6843,
  "joint_4" : -0.0118,
  "joint_5" : -0.1,
  "joint_6" : 0.0, 
  "position": [-0.156, 0.527, -0.32077],     # x,y,z
  "orientation": [-0.7712, 0.5591, 0.2416, 0.185], 
  },

  # 自动标定初始位置
  "arm_auto_init":{
    "joint_1" : 0.0,
    "joint_2" : 0.1,
    "joint_3" : 0.9,
    "joint_4" : 0.0,
    "joint_5" : 0.8852,
    "joint_6" : 0.0, 
  }
}

# 自动标定限制量
Auto_calibration_limit = {
  "on_desk": {
    "joint_1" : [-0.8, 0.8],
    "joint_2" : [-0.8, 0.8],
    "joint_3" : [0.6, 1.3],
    "joint_4" : [-0.8, 0.8],
    "joint_5" : [0.0, 0.8852],
    "joint_6" : [-0.8, 0.8], 
  },
  "on_weiyi": {
    "joint_1" : [-2.4, -0.8],
    "joint_2" : [-1.7, 1.7],       # [1.85+1.1+0.2, 1.7+0.83+0.4, 1.0+2.2-0.6, 1.0+1.5+1.5]
    "joint_3" : [0.6, 1.3],        # [3.15, 2.93, 2.6, 4.0]
    "joint_4" : [-0.8, 0.8],
    "joint_5" : [0.0, 0.8852],
    "joint_6" : [-0.8, 0.8], 
  }
}
 
Auto_calibration_sum_rad = {
  "on_desk": 1.8852,
  "on_weiyi": 2.6,
} 
Joint_name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

