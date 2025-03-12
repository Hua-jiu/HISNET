# HISNET-Supplement-Code

## Description

This repository contains the supplementary code for the paper "*Sky islands of Southwest China. II: Unraveling hidden species diversity of talpid moles using phylogenomics and skull-based deep learning*". The code is organized into two main folders, **Model_Train** and **HISNet_Train**, each serving distinct purposes in the context of the research.

- **Model_Train**: This folder includes scripts to train various baseline models on the dataset, and the scripts for training the HISNET model (including species classifiers for each species).
- **Model_Test**: This folder focuses on testing the HISNET model we have trained on the dataset and evaluating its performance. You can also train you owe model and replace the weight file in the **Model_Test** folder to test your model.

---

## Folder Structure
```
Suplement_Code/
├── Model_Test/
│   ├── data/
│   │   ├── genus_labels.json
│   │   ├── more_species_labels.json
│   │   └── species_labels.json
│   ├── docs/
│   │   └── Consequence.txt
│   ├── species_classfier/
│   │   ├── data/
│   │   │   └── more_species_labels.json
│   │   └── weights/
│   │       ├── EB3_Euroscaptor/
│   │       │   └── best_network.pth
│   │       ├── EB3_Mogera/
│   │       │   └── best_network.pth
│   │       ├── EB3_Parascaptor/
│   │       │   └── best_network.pth
│   │       ├── EB3_Scapanus/
│   │       │   └── best_network.pth
│   │       ├── EB3_Scaptonyx/
│   │       │   └── best_network.pth
│   │       ├── EB3_Talpa/
│   │       │   └── best_network.pth
│   │       └── EB3_Uropsilus/
│   │           └── best_network.pth
│   ├── tools/
│   │   ├── Early_Stopping.py
│   │   ├── file_utils.py
│   │   ├── get_sample_predict.py
│   │   ├── GPU_Detecter.py
│   │   └── SpeciesClassfier_ind.py
│   ├── weights/
│   │   └── EfficientNet-B3/
│   │       └── best_network.pth
│   └── predict_ind_ToSpecies.py
├── Model_Train/
│   ├── Baseline_Model_Compare_Train/
│   │   ├── data/
│   │   │   ├── test/
│   │   │   ├── train/
│   │   │   ├── genus_labels.json
│   │   │   ├── more_species_labels.json
│   │   │   └── species_labels.json
│   │   ├── docs/
│   │   │   └── Consequence.txt
│   │   ├── logs/
│   │   │   └── log.txt
│   │   ├── tools/
│   │   │   ├── Early_Stopping.py
│   │   │   ├── extract_bestAcc.py
│   │   │   ├── file_utils.py
│   │   │   ├── get_par.py
│   │   │   ├── get_scripts_name.py
│   │   │   └── GPU_Detecter.py
│   │   ├── weights/
│   │   ├── AlexNet.py
│   │   ├── EfficientNet-B0.py
│   │   ├── EfficientNet-B2.py
│   │   ├── EfficientNet-B3.py
│   │   ├── EfficientNet-B4.py
│   │   ├── GoogleNet.py
│   │   ├── MobileNet-V2.py
│   │   ├── MobileNet-V3Large.py
│   │   ├── MobileNet-V3Small.py
│   │   ├── ResNet_152.py
│   │   ├── ResNet_18.py
│   │   ├── ResNet_34.py
│   │   ├── ShuffleNet-V2_05.py
│   │   ├── ShuffleNet-V2_10.py
│   │   ├── ShuffleNet-V2_15.py
│   │   ├── ShuffleNet-V2_20.py
│   │   ├── VGGNet_11.py
│   │   ├── VGGNet_16.py
│   │   └── VGGNet_19.py
│   └── HISNet_Train/
│       ├── Model_Train/
│       │   ├── data/
│       │   │   ├── test/
│       │   │   ├── train/
│       │   │   ├── genus_labels.json
│       │   │   ├── more_species_labels.json
│       │   │   └── species_labels.json
│       │   ├── docs/
│       │   │   └── Consequence.txt
│       │   ├── logs/
│       │   │   └── log.txt
│       │   ├── tools/
│       │   │   ├── Early_Stopping.py
│       │   │   ├── extract_bestAcc.py
│       │   │   ├── file_utils.py
│       │   │   ├── get_par.py
│       │   │   ├── get_scripts_name.py
│       │   │   └── GPU_Detecter.py
│       │   ├── weights/
│       │   └── EfficientNet-B3.py
│       └── Species_Classfier_Train/
│           ├── data/
│           │   ├── test/
│           │   ├── train/
│           │   └── more_species_labels.json
│           ├── docs/
│           ├── logs/
│           ├── tools/
│           │   ├── Early_Stopping.py
│           │   ├── extract_bestAcc.py
│           │   ├── file_utils.py
│           │   ├── generate_json_list.py
│           │   ├── get_par.py
│           │   ├── get_scripts_name.py
│           │   └── GPU_Detecter.py
│           ├── weights/
│           ├── EfficientNet_B3_Euroscaptor.py
│           ├── EfficientNet_B3_Mogera.py
│           ├── EfficientNet_B3_Parascaptor.py
│           ├── EfficientNet_B3_Scapanus.py
│           ├── EfficientNet_B3_Scaptonyx.py
│           ├── EfficientNet_B3_Talpa.py
│           └── EfficientNet_B3_Uropsilus.py
├── file_tree.py
├── README.md
└── requirements.txt
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hua-jiu/HISNET.git
   ```

2. **Install required packages**:
   Ensure you have Python (version 3.10 or higher) and the necessary libraries installed. You can create a virtual environment and install the required packages using `requirements.txt`:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
## How to Use

1. **Baseline Model Comparing**:
    Navigate to the `Model_Train` directory and execute the desired script to train a specific baseline model. For example:
    ```bash
    python AlexNet.py
    ```
    Before you run the script, make sure the dataset is in the folder `/data/` and have been split into train and test sets.

2. **HISNET Train**:
    In the `HISNet_Train` directory, you can find the scripts for training the EfficientNet-B3 model and species classifiers. To train the EfficientNet-B3 model, navigate to `HISNet_Train/Model_Train` and run:
    ```bash
    python EfficientNet-B3.py
    ```
    To train the species classifiers, navigate to `HISNet_Train/Species_Classifier_Train` and run the corresponding script for each species.

3. **HISNET Test**:
    To test the HISNET model, navigate to the `HISNet_Test` directory and run:
     ```bash
     python predict_ind_ToSpecies.py
     ```
    This script will load the trained EfficientNet-B3 model and species classifiers, and predict the species for each specimen in the test dataset. The results will be saved in the `docs` folder.
    Before you run the script, make sure the test dataset is in the folder `/data/test`.
