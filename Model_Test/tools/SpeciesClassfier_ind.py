import json
import os
import torch
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
import torchvision.models as models
from get_sample_predict import get_sample_predict


class SpeceseClassifier:
    def __init__(self, device, data_dir):
        self.device = device
        print(f"{device}")
        json_file = open(f"{data_dir}/data/more_species_labels.json", "r")
        class_indict = json.load(json_file)

        # 各种级分类器的分类数
        self.model1_Euroscaptor_num = len(class_indict["Euroscaptor"])
        self.model2_Mogera_num = len(class_indict["Mogera"])
        self.model3_Parascaptor_num = len(class_indict["Parascaptor"])
        self.model4_Scapanus_num = len(class_indict["Scapanus"])
        self.model5_Scaptonyx_num = len(class_indict["Scaptonyx"])
        self.model6_Talpa_num = len(class_indict["Talpa"])
        self.model7_Uropsilus_num = len(class_indict["Uropsilus"])

        # 各种级分类器的权重参数文件
        self.data_dir = data_dir
        self.weights_Euroscaptor_path = (
            f"{self.data_dir}/weights/EB3_Euroscaptor/best_network.pth"
        )
        self.weights_Mogera_path = (
            f"{self.data_dir}/weights/EB3_Mogera/best_network.pth"
        )
        self.weights_Parascaptor_path = (
            f"{self.data_dir}/weights/EB3_Parascaptor/best_network.pth"
        )
        self.weights_Scapanus_path = (
            f"{self.data_dir}/weights/EB3_Scapanus/best_network.pth"
        )
        self.weights_Scaptonyx_path = (
            f"{self.data_dir}/weights/EB3_Scaptonyx/best_network.pth"
        )
        self.weights_Talpa_path = f"{self.data_dir}/weights/EB3_Talpa/best_network.pth"
        self.weights_Uropsilus_path = (
            f"{self.data_dir}/weights/EB3_Uropsilus/best_network.pth"
        )

        # 初始化模型,载入权重文件
        self.model_init()

    def model_init(self):
        # create model
        model1_Euroscaptor = models.efficientnet_b3()
        model2_Mogera = models.efficientnet_b3()
        model3_Parascaptor = models.efficientnet_b3()
        model4_Scapanus = models.efficientnet_b3()
        model5_Scaptonyx = models.efficientnet_b3()
        model6_Talpa = models.efficientnet_b3()
        model7_Uropsilus = models.efficientnet_b3()

        model1_Euroscaptor.classifier[1] = torch.nn.Linear(
            model1_Euroscaptor.classifier[1].in_features, self.model1_Euroscaptor_num
        )
        model2_Mogera.classifier[1] = torch.nn.Linear(
            model2_Mogera.classifier[1].in_features, self.model2_Mogera_num
        )
        model3_Parascaptor.classifier[1] = torch.nn.Linear(
            model3_Parascaptor.classifier[1].in_features, self.model3_Parascaptor_num
        )
        model4_Scapanus.classifier[1] = torch.nn.Linear(
            model4_Scapanus.classifier[1].in_features, self.model4_Scapanus_num
        )
        model5_Scaptonyx.classifier[1] = torch.nn.Linear(
            model5_Scaptonyx.classifier[1].in_features, self.model5_Scaptonyx_num
        )
        model6_Talpa.classifier[1] = torch.nn.Linear(
            model6_Talpa.classifier[1].in_features, self.model6_Talpa_num
        )
        model7_Uropsilus.classifier[1] = torch.nn.Linear(
            model7_Uropsilus.classifier[1].in_features, self.model7_Uropsilus_num
        )

        self.model1_Euroscaptor = model1_Euroscaptor.to(self.device)
        self.model2_Mogera = model2_Mogera.to(self.device)
        self.model3_Parascaptor = model3_Parascaptor.to(self.device)
        self.model4_Scapanus = model4_Scapanus.to(self.device)
        self.model5_Scaptonyx = model5_Scaptonyx.to(self.device)
        self.model6_Talpa = model6_Talpa.to(self.device)
        self.model7_Uropsilus = model7_Uropsilus.to(self.device)

        # load model weights
        weights_Euroscaptor_path = f"{self.weights_Euroscaptor_path}"
        assert os.path.exists(
            weights_Euroscaptor_path
        ), "file: '{}' dose not exist.".format(weights_Euroscaptor_path)
        model1_Euroscaptor.load_state_dict(
            torch.load(weights_Euroscaptor_path, map_location=self.device)
        )

        weights_Mogera_path = f"{self.weights_Mogera_path}"
        assert os.path.exists(weights_Mogera_path), "file: '{}' dose not exist.".format(
            weights_Mogera_path
        )
        model2_Mogera.load_state_dict(
            torch.load(weights_Mogera_path, map_location=self.device)
        )

        weights_Parascaptor_path = f"{self.weights_Parascaptor_path}"
        assert os.path.exists(
            weights_Parascaptor_path
        ), "file: '{}' dose not exist.".format(weights_Parascaptor_path)
        model3_Parascaptor.load_state_dict(
            torch.load(weights_Parascaptor_path, map_location=self.device)
        )

        weights_Scapanus_path = f"{self.weights_Scapanus_path}"
        assert os.path.exists(
            weights_Scapanus_path
        ), "file: '{}' dose not exist.".format(weights_Scapanus_path)
        model4_Scapanus.load_state_dict(
            torch.load(weights_Scapanus_path, map_location=self.device)
        )

        weights_Scaptonyx_path = f"{self.weights_Scaptonyx_path}"
        assert os.path.exists(
            weights_Scaptonyx_path
        ), "file: '{}' dose not exist.".format(weights_Scaptonyx_path)
        model5_Scaptonyx.load_state_dict(
            torch.load(weights_Scaptonyx_path, map_location=self.device)
        )

        weights_Talpa_path = f"{self.weights_Talpa_path}"
        assert os.path.exists(weights_Talpa_path), "file: '{}' dose not exist.".format(
            weights_Talpa_path
        )
        model6_Talpa.load_state_dict(
            torch.load(weights_Talpa_path, map_location=self.device)
        )

        weights_Uropsilus_path = f"{self.weights_Uropsilus_path}"
        assert os.path.exists(
            weights_Uropsilus_path
        ), "file: '{}' dose not exist.".format(weights_Uropsilus_path)
        model7_Uropsilus.load_state_dict(
            torch.load(weights_Uropsilus_path, map_location=self.device)
        )

        self.model1_Euroscaptor.eval()
        self.model2_Mogera.eval()
        self.model3_Parascaptor.eval()
        self.model4_Scapanus.eval()
        self.model5_Scaptonyx.eval()
        self.model6_Talpa.eval()
        self.model7_Uropsilus.eval()

    def predict(self, GenusPredict_dict):
        """get a dict which the jpgPath as the key and the predict species name as the value

        Args:
            GenusPredict_dict (_type_): key is genus name, value of the key is samples path
        """
        json_file = open(f"{self.data_dir}/data/more_species_labels.json", "r")
        species_label_dict = json.load(json_file)
        predict_dict = {}

        for genus_name in GenusPredict_dict.keys():
            bpPredicted_label = genus_name
            ind_PathList = GenusPredict_dict[genus_name]
            for ind_Path in ind_PathList:
                with torch.no_grad():
                    # predict class
                    if bpPredicted_label == "Euroscaptor":
                        output = get_sample_predict(
                            ind_Path,
                            self.model1_Euroscaptor,
                            self.device,
                            self.model1_Euroscaptor_num,
                        )
                    elif bpPredicted_label == "Mogera":
                        output = get_sample_predict(
                            ind_Path,
                            self.model2_Mogera,
                            self.device,
                            self.model2_Mogera_num,
                        )
                    elif bpPredicted_label == "Parascaptor":
                        output = get_sample_predict(
                            ind_Path,
                            self.model3_Parascaptor,
                            self.device,
                            self.model3_Parascaptor_num,
                        )
                    elif bpPredicted_label == "Scapanus":
                        output = get_sample_predict(
                            ind_Path,
                            self.model4_Scapanus,
                            self.device,
                            self.model4_Scapanus_num,
                        )
                    elif bpPredicted_label == "Scaptonyx":
                        output = get_sample_predict(
                            ind_Path,
                            self.model5_Scaptonyx,
                            self.device,
                            self.model5_Scaptonyx_num,
                        )
                    elif bpPredicted_label == "Talpa":
                        output = get_sample_predict(
                            ind_Path,
                            self.model6_Talpa,
                            self.device,
                            self.model6_Talpa_num,
                        )
                    elif bpPredicted_label == "Uropsilus":
                        output = get_sample_predict(
                            ind_Path,
                            self.model7_Uropsilus,
                            self.device,
                            self.model7_Uropsilus_num,
                        )
                    else:
                        predict_dict[ind_Path] = bpPredicted_label
                        continue
                    predict_name = species_label_dict[bpPredicted_label][int(output)]
                    predict_dict[ind_Path] = predict_name
                    # print(predict_name)
        return predict_dict
