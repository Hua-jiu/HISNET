import os

def find_python_scripts(directory):
    # List all files and directories under the given path
    files_and_directories = os.listdir(directory)

    # Filter all files ending with .py
    python_scripts = [f for f in files_and_directories if f.endswith('.py')]

    return python_scripts

# Use current folder as the directory
current_directory = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_directory)
scripts = find_python_scripts(root_dir)

print("Python script files:")
for script in scripts:
    print(f"'{script}',")
