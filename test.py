import json
import numpy as np

# 读取 JSON 文件
with open('hist_stats.json', 'r') as file:
    data = json.load(file)

# 对 stats 下的每一个键的列表进行归一化
for key, values in data['stats'].items():
    total = sum(values)
    if total != 0:
        normalized_values = [value / total for value in values]
    else:
        normalized_values = values  # 如果总和为0，保持原值
    data['stats'][key] = normalized_values

# 将归一化后的数据写回到 JSON 文件
with open('hist_stats1.json', 'w') as file:
    json.dump(data, file, indent=4)

print("归一化处理完成，并已写回到 JSON 文件。")