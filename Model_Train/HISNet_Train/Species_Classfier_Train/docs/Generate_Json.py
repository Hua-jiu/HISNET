# genrate json file
import os
import json

# change the path to your folder
folder_path = ''
test_dir = f'{folder_path}/data/test'

# read all files in the directory
items = os.listdir(test_dir)
items.sort()

# create a dictionary with the index as the key and the file name as the value
mapping_dict = {int(i): item for i, item in enumerate(items)}

# translate the dictionary to JSON format
json_mapping = json.dumps(mapping_dict, indent=4)

# save the JSON file
print(json_mapping)
with open(f'{folder_path}/data/genus_labels.json', 'w') as json_file:
        json_file.write(json_mapping)
