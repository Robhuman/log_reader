
from datetime import datetime
import re

# 给定的文本数据

file_path = "D://Rvbust//20240515+_Operationlog (1).txt"
with open(file_path, 'r') as file:
    data = file.read()

from datetime import datetime
import re



# 提取所有时间数据
time_data = re.findall(r'\d{2}:\d{2}:\d{2}:\d{3}', data)

if len(time_data) >= 2:
    start_time_str = "1900-01-01 " + time_data[0]
    end_time_str = "1900-01-01 " + time_data[-1]

    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S:%f")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S:%f")

    # 计算时间差
    time_diff = end_time - start_time

    print("起始时间：", start_time)
    print("结束时间：", end_time)
    print("时间差：", time_diff)
else:
    print("未找到足够的时间数据")


# 提取所有计算时间数据
time_data = re.findall(r'计算后发给机器人信息时间：([\d.]+)秒', data)



# 提取"PickBatteryRight"的数量
pick_battery_count = len(re.findall(r'<RobData><STR>PickBatteryRight</STR>', data)) * 5

# 提取"计算后发给机器人信息时间"的数量
time_count = len(re.findall(r'计算后发给机器人信息时间', data))

print("提取的PickBatteryRight数量：", pick_battery_count)
print("提取的计算后发给机器人信息时间数量：", time_count)

# 计算平均值
if time_data:
    time_data = [float(time) for time in time_data]
    average_time = sum(time_data) / len(time_data)
    print("提取的计算时间数据：", time_data)
    print("平均值：", average_time)
else:
    print("未找到计算时间数据")
