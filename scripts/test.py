import os
import json

# 指定文件夹路径
folder_path = 'datasets/GaitData/train'

# 获取文件夹下的所有目录名字
dir_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

# 生成 JSON 文件
json_data = {
    "TRAIN_SET": dir_names
}

# 写入 JSON 文件
with open('output.json', 'w') as f:
    json.dump(json_data, f, indent=4)