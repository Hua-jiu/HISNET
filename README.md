# Sky-islands-of-Southwest-China.-II-Supplement-Code

## Description

This repository contains the supplementary code for the paper "*Sky islands of Southwest China. II: Unraveling hidden species diversity of talpid moles using phylogenomics and skull-based deep learning*". The code is organized into two main folders, **Model_Train** and **HISNet_Train**, each serving distinct purposes in the context of the research.

- **Model_Train**: This folder includes scripts to train various baseline models on the dataset.
- **HISNet_Train**: This folder focuses on training the EfficientNet-B3 model and species classifiers for different species.

---

## Folder Structure
Sky-Islands-Supplement-Code/
│
├── Model_Train/
│   ├── AlexNet.py
│   ├── EfficientNet-B0.py
│   ├── EfficientNet-B2.py
│   ├── EfficientNet-B3.py
│   ├── EfficientNet-B4.py
│   ├── GoogleNet.py
│   ├── MobileNet-V2.py
│   ├── MobileNet-V3Large.py
│   ├── MobileNet-V3Small.py
│   ├── ResNet_18.py
│   ├── ResNet_34.py
│   ├── ResNet_152.py
│   ├── ShuffleNet-V2_05.py
│   ├── ShuffleNet-V2_10.py
│   ├── ShuffleNet-V2_15.py
│   ├── ShuffleNet-V2_20.py
│   ├── VGGNet_11.py
│   ├── VGGNet_16.py
│   ├── VGGNet_19.py
│   └── tools/
│
├── HISNet_Train/
│   ├── Model_Train/
│   │   └── EfficientNet-B3.py
│   ├── Species_Classifier_Train/
│   │   ├── EfficientNet_B3_Euroscaptor.py
│   │   ├── EfficientNet_B3_Mogera.py
│   │   ├── EfficientNet_B3_Parascaptor.py
│   │   ├── EfficientNet_B3_Scapanus.py
│   │   ├── EfficientNet_B3_Scaptonyx.py
│   │   ├── EfficientNet_B3_Talpa.py
│   │   └── EfficientNet_B3_Uropsilus.py
│
└── README.md
│
└── requirements.txt


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hua-jiu/Sky-islands-of-Southwest-China.-II-Supplement-Code.git
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

2. **HISNet Training**:
    In the `HISNet_Train` directory, you can find the scripts for training the EfficientNet-B3 model and species classifiers. To train the EfficientNet-B3 model, navigate to `HISNet_Train/Model_Train` and run:
    ```bash
    python EfficientNet-B3.py
    ```
    To train the species classifiers, navigate to `HISNet_Train/Species_Classifier_Train` and run the corresponding script for each species.
