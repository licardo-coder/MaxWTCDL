# 导出数据
import pandas as pd
# 将ndarray转换为DataFrame
# sensor_xx_yy_x_y_radius=50
sensor_xx_yy_x_y_radius = (sensor_x, sensor_y)
df = pd.DataFrame(sensor_xx_yy_x_y_radius)
df.to_excel('sensor_xx_yy_x_y_radius.xlsx', index=False, header=True)  # header=False表示不写入列名

target_xyw = (target_x, target_y, target_w)
df = pd.DataFrame(target_xyw)
df.to_excel('target_xy.xlsx', index=False, header=True)  # header=False表示不写入列名

action_cost_reward = (action_sensor_move, distance_cost, up_reward)
df = pd.DataFrame(action_cost_reward)
df.to_excel('action_cost_reward.xlsx', index=False, header=True)  # header=False表示不写入列名


# 读取 target_xy.xlsx 文件
target_xyw_df = pd.read_excel('target_xy.xlsx')

target_xyw = target_xyw_df.to_numpy()
target_x = target_xyw[0, :]
target_y = target_xyw[1, :]
target_w = target_xyw[2, :]

snesor_x_y_df = pd.read_excel('sensor_xx_yy_x_y_radius.xlsx')
snesor_x_y = snesor_x_y_df.to_numpy()
sensor_x = snesor_x_y[0, :]
sensor_y = snesor_x_y[1, :]



