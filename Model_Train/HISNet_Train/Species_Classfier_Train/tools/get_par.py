import os
import ast
import pandas as pd

# Get all Python scripts in the specified directory
def get_python_scripts(directory):
    scripts = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            scripts.append(os.path.join(directory, filename))
    return scripts

# Parse parameters from the Python script
def extract_params_from_script(script_path):
    with open(script_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=script_path)

    params = {"batch_size": None, "learning_rate": None, "step_size": None}
    for node in ast.walk(tree):
        # Only capture assignment parameters
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in params:
                        # Get the concrete parameter value
                        if isinstance(node.value, ast.Constant):
                            params[target.id] = node.value.value
                        elif isinstance(node.value, ast.Num):  # Compatible with older Python versions
                            params[target.id] = node.value.n
    return params

# Collect parameters from all scripts
def get_model_params(directory):
    scripts = get_python_scripts(directory)
    model_data = []

    for script in scripts:
        model_name = os.path.basename(script).replace(".py", "")
        params = extract_params_from_script(script)
        params["model_name"] = model_name
        model_data.append(params)

    return pd.DataFrame(model_data)

# Plot parameter table
def plot_model_params(df, directory):
    print(df)
    # Save as Excel file
    df.to_excel(f"{directory}/docs/model_params.xlsx", index=False)

# Specify the directory where your model scripts are stored
directory = "/mnt/storage-data2/anlong/MoleProject/New_data_Exp_20240410/Other_net_compare/To-Genus/Data_224"

# Get the model parameter table
df = get_model_params(directory)

# Plot or save the parameter table
plot_model_params(df, directory)