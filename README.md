# HISNET-Supplement-Code

## Description

This repository contains the supplementary code for the paper "*Sky islands of Southwest China. II: Unraveling hidden species diversity of talpid moles using phylogenomics and skull-based deep learning*". The code is organized into two main folders, **Model_Train** and **HISNET_Train**, each serving distinct purposes in the context of the research.

- **Model_Train**: This folder includes scripts to train various baseline models on the dataset, and the scripts for training the HISNET model (including species classifiers for each species).
- **Model_Test**: This folder focuses on testing the HISNET model we have trained on the dataset and evaluating its performance. You can also train you owe model and replace the weight file in the **Model_Test** folder to test your model.

---

## Folder Structure

```
Suplement_Code/
├── Model_Test/
│   ├── data/
│   │   ├── test/
│   │   ├── genus_labels.json
│   │   ├── more_species_labels.json
│   │   └── species_labels.json
│   ├── docs/
│   │   └── Consequence.txt
│   ├── species_classfier/
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
│   │   │   ├── Generate_Json.py
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
│   └── HISNET_Train/
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
│           │   └── Generate_Json.py
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
├── Quick_Start/
│   ├── data/
│   │   ├── genus_labels.json
│   │   ├── more_species_labels.json
│   │   └── species_labels.json
│   ├── docs/
│   ├── species_classfier/
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
│   └── Quick_Start.py
├── project.toml
└── README.md
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hua-jiu/HISNET.git
   cd HISNET
   ```

2. **Install required packages**:
   Ensure that you have installed uv on your Linux system (https://docs.astral.sh/uv/getting-started/installation). HISNET requires Python >= 3.10.
   ```bash
   # Create and activate a virtual environment with Python 3.10
   uv venv --python=3.10
   source .venv/bin/activate

   # Install dependencies from project.toml
   uv pip install -e .
   ```
## Quick Start

1. **Navigating to the directory**:
    Open your terminal application. Use the `cd` (change directory) command to move into the `Quick_Start` directory. For example, if the `HISNET` directory is located on your home directory, you can run the following command:
    ```bash
    cd home/HISNET/Quick_Start
    # Replace `home/` with the actual path to the `Quick_Start` directory on your machine.
    ```

2. **Run the python script**:
    Once you are in the `Quick_Start` directory, execute the `Quick_Start.py` script by running the following command in the terminal:
    ```bash
    python Quick_Start.py
    ```

    This Python script is specifically designed to initialize and run the HISNET model. When you execute it, the model will start processing, analyzing the input data related to talpidae mole samples, and generate identification results.

    After running the script, you can expect to see the identification results of different talpidae mole samples in the `./Quick_Start/docs`.

## Get your own model

#### Baseline Model Comparing

1. **Preparing the Dataset**:
    Before training, ensure that your dataset is placed in the `/data/` folder relative to the current working directory. Each image in the dataset should be isolated with a size of **224x224** pixels. Additionally, split the dataset into separate train and test sets in an appropriate ratio (e.g., 80% for training and 20% for testing). This split is essential for accurately assessing the model's generalization ability during training.

    To ensure the subsequent testing scripts run correctly, please make sure that the image filenames follow the naming convention:
    **`genus_label#species_label#individual_label#image_number#sample_type#sample_view`**.
    You can also find a naming example in the `HISNET/Quick_Start/data/` folder.

2. **Select and Run a Baseline Model Script**
   In the `Model_Train/Baseline_Model_Compare_Train` directory, you'll find various scripts for different baseline models. To start training a specific model, such as AlexNet, execute the corresponding Python script using the following command:
    ```bash
    python AlexNet.py
    ```
    Available scripts may include other common architectures like VGG, ResNet, etc. You can choose the model based on your dataset characteristics and performance expectations.

3. **Monitoring and Saving Results**
   + **Training Logs**: The training process details, including metrics like loss and accuracy at each epoch, will be logged in the `log.txt` file within the `logs` directory. You can review this file to track the progress and performance of the training.
   + **Training Results**: The final evaluation results of the trained model on the test set, such as precision, recall, and F1-score, will be saved in the `docs` directory.
   + **Model Weights**: The learned weights of the trained model will be stored in the `weights` directory. These weights can be used for further inference or model analysis.

4. **Exploring Alternative Models**
   If the provided network model scripts don't yield satisfactory performance on your dataset, don't worry! You have the flexibility to explore and implement other network models. There are numerous open-source deep learning architectures available on platforms like GitHub and Papers with Code. You can search for architectures that are suitable for your dataset type (e.g., image classification, object detection) and adapt them for training in this project. You may need to adjust the code structure, input/output formats, and hyperparameters according to the requirements of the new model, but this gives you the opportunity to potentially achieve better results tailored to your specific dataset.

#### Train species classifiers

1. **Dataset Preparation**:
    Before starting the training process, ensure that your dataset is correctly placed in the `/data/` folder relative to the working directory. The image size and format should match those used during the baseline model training to avoid compatibility issues. Additionally, organize your dataset by following the "genus/species" hierarchy. That is, create subfolders for each genus within the main dataset folder, and then further create subfolders for individual species within each genus folder. This structure enables the training scripts to correctly associate and train species classifiers for different genera, ensuring accurate classification during the training process. Also, confirm that the dataset has been properly partitioned into training and testing subsets, typically in a ratio of 80% for training and 20% for testing. This division is essential for evaluating the performance and generalization ability of the species classifiers.

2. **Label Configuration**
   The `HISNET_Train/Species_Classifier_Train/data/more_species_labels.json` file stores the labels for the species classifiers. If you plan to train a new species classifier, you must update the labels in this file. Here are the steps to do so:

    + Open the `more_species_labels.json` file using a text editor.

    + Modify the label values to accurately represent the new species you're adding. Ensure that the JSON format remains intact, with proper key-value pairs.

    + Save the file before starting the training process for the new species classifier. This ensures that the model is trained with the correct classification targets.

3. **Training Species Classifiers**:
   The `HISNET_Train/Species_Classifier_Train` directory contains a set of training scripts designed to serve as a foundational example for training species classifiers. These scripts are not intended as a one - size - fits - all solution but rather as a starting point for you to build upon and customize according to your specific requirements.
   + **Customization**: A common adjustment might be necessary when using a different baseline model. In such cases, you'll need to update the model loading section within the script to ensure compatibility with the new baseline. This typically involves changing the import statements, model initialization parameters, and any associated pre - processing steps related to the model.
   + **Genus - Specific Scripts**: Each genus has its own dedicated training script. This design allows for targeted training and optimization based on the characteristics of different genera. When working with multiple genera, carefully select and run the appropriate script for each one. Pay attention to the naming convention of the scripts, as it usually indicates the genus or species the script is intended for. For example, scripts might be named in a format like `ModelName_GenusName.py`.


#### HISNET Test
1. **Load the weights and data**
    To test the HISNET model, navigate to the `Model_Test` directory and place your baseline model weights into the `weights` sub-folder. Next, copy your entire species classifier file into the `Model_Test/species_classfier` directory.
    Before you run the script, make sure the test dataset is in the folder `/data/test`.

2. **Start classify**
    The `predict_ind_ToSpecies.py` script is the main entry point for the classification process. Its primary function is to load the trained EfficientNet-B3 model and the species classifiers. However, if you are using a different baseline model, you must modify the model loading section of the script. This usually involves changing the import statements related to the model, adjusting the model initialization code, and ensuring that any required model - specific configurations are set correctly. For example, if you are using a ResNet model instead of EfficientNet-B3, you may need to change lines like:
    ```python
    model = torchvision.models.efficientnet_b3()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 18)
    model = model.to(device)
    weights_path = f"{data_dir}/weights/EfficientNet-B3/best_network.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    ```
    to somethig like:
    ```python
    model = torchvision.models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, num_class)
    model = model.to(device)
    weights_path = f"{data_dir}/weights/ResNet_34/best_network.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    ```

3. **Classification Process**
   + **Genus - Level Classification**: When you run the `predict_ind_ToSpecies.py` script, the data will first be classified at the genus level using the EfficientNet-B3 model (or your modified baseline model). The model processes the input images and generates genus - level predictions based on the learned patterns in the training data.

   + **Specimen - Level Prediction**: The results of the genus-level classification are then passed to the `/tools/``get_sample_predict.py` script. This script processes the genus - level predictions and assigns the same genus prediction label to every image within each specimen. This step aggregates the individual image predictions to the specimen level, providing a more comprehensive view of the classification for each sample.
   For your own project, you can change the weight of each image view contribute to the final specimens prediction in the `get_sample_predict.py` script. You may need to change lines like:
   ```python
   def get_sample_predict(sample_dir, model, device, class_num):
       # Statistical accuracy for upper and lower jaw surfaces
       s_l_acc = 0.9589
       s_d_acc = 0.9710
       s_v_acc = 0.9728
       m_d_acc = 0.9503
   ```
   to something like:
   ```python
   def get_sample_predict(sample_dir, model, device, class_num):
       # Statistical accuracy for upper and lower jaw surfaces
       s_l_acc = "your skull lateral surface accuracy"
       s_d_acc = "your skull dorsal surface accuracy"
       s_v_acc = "your skull ventral surface accuracy"
       m_d_acc = "your mandible dorsal surface accuracy"
   ```

   + **Species - Level Classification**: Finally, the species classifiers are utilized to predict the species for each individual specimen. These classifiers take the specimen - level genus information and further analyze the data to determine the specific species. The results of these species-level predictions are saved in the `docs` folder for easy access and review. You can find detailed prediction reports, accuracy metrics, and other relevant information in this folder, which can be used for further analysis and evaluation of the HISNET model's performance.