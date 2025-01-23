import os

def find_python_scripts(directory):
    # 获取指定目录下的所有文件和文件夹名称
    files_and_directories = os.listdir(directory)

    # 筛选出所有以.py结尾的文件
    python_scripts = [f for f in files_and_directories if f.endswith('.py')]

    return python_scripts

# 使用当前文件夹作为目录
current_directory = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_directory)
scripts = find_python_scripts(root_dir)

print("Python 脚本文件：")
for script in scripts:
    print(f"'{script}',")
