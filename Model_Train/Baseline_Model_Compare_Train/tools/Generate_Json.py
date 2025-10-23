# Generate JSON dictionary
import os
import json

# Replace with your folder path
folder_path = '/mnt/storage-data2/anlong/MoleProject/New_data_Exp_20240410/CompleteIndTest/DATABASE'
test_dir = f'{folder_path}/data1/test'

# Read subfolders or files in the folder
items = os.listdir(test_dir)
items.sort()

# Create mapping dictionary
mapping_dict = {int(i): item for i, item in enumerate(items)}

# Convert dictionary to JSON format
json_mapping = json.dumps(mapping_dict, indent=4)

# Print or save JSON mapping
print(json_mapping)
with open(f'{folder_path}/data1/genus_labels.json', 'w') as json_file:
        json_file.write(json_mapping)
