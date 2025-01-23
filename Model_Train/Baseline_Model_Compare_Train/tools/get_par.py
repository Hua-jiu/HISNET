import os
import ast
import pandas as pd

# 获取指定目录下的所有 Python 脚本
def get_python_scripts(directory):
    scripts = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            scripts.append(os.path.join(directory, filename))
    return scripts

# 解析 Python 脚本中的参数
def extract_params_from_script(script_path):
    with open(script_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=script_path)

    params = {"batch_size": None, "learning_rate": None, "step_size": None}
    for node in ast.walk(tree):
        # 只获取赋值操作的参数
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in params:
                        # 获取具体的参数值
                        if isinstance(node.value, ast.Constant):
                            params[target.id] = node.value.value
                        elif isinstance(node.value, ast.Num):  # 兼容旧的 python 版本
                            params[target.id] = node.value.n
    return params

# 获取所有脚本中的参数
def get_model_params(directory):
    scripts = get_python_scripts(directory)
    model_data = []

    for script in scripts:
        model_name = os.path.basename(script).replace(".py", "")
        params = extract_params_from_script(script)
        params["model_name"] = model_name
        model_data.append(params)

    return pd.DataFrame(model_data)

# 绘制参数表
def plot_model_params(df, directory):
    print(df)
    # 如果需要保存为 CSV 文件
    df.to_excel(f"{directory}/docs/model_params.xlsx", index=False)

# 指定你存放模型脚本的目录
directory = "/mnt/storage-data2/anlong/MoleProject/New_data_Exp_20240410/Other_net_compare/To-Genus/Data_224"

# 获取模型参数表
df = get_model_params(directory)

# 绘制或保存参数表
plot_model_params(df, directory)