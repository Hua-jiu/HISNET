# 生成 json 字典
import os
import json

# 替换为你的文件夹路径
folder_path = '/mnt/storage-data2/anlong/MoleProject/New_data_Exp_20240410/CompleteIndTest/DATABASE'
test_dir = f'{folder_path}/data1/test'

# 读取文件夹中的子文件夹或文件
items = os.listdir(test_dir)
items.sort()

# 创建映射字典
mapping_dict = {int(i): item for i, item in enumerate(items)}

# 将字典转换为 JSON 格式
json_mapping = json.dumps(mapping_dict, indent=4)

# 打印或保存 JSON 映射
print(json_mapping)
with open(f'{folder_path}/data1/genus_labels.json', 'w') as json_file:
        json_file.write(json_mapping)
