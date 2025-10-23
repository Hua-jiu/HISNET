import torch
from torchvision import transforms
import os
from math import exp
from PIL import Image
import file_utils as ut
import numpy as np
import math


def get_sample_predict(sample_dir, model, device, class_num):
    # Per-view accuracy for maxilla and mandible
    s_l_acc = 0.9589
    s_d_acc = 0.9710
    s_v_acc = 0.9728
    m_d_acc = 0.9503

    # model = torchvision.models.efficientnet_b0()

    weight_average_predict = [[]]
    for i in range(class_num):
        weight_average_predict[0].append(0)
    # weight_average_predict.append()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Iterate through individual folder
    samples = os.listdir(sample_dir)
    images = samples
    output = []
    weight_average_predict = [[]]
    for i in range(class_num):
        weight_average_predict[0].append(0)
    # First pass: iterate all sample images for one individual
    for image in images:
        image_path = os.path.join(sample_dir, image)
        img = Image.open(image_path)
        img = transform(img)

        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output_pre = torch.squeeze(model(img.to(device))).cpu()
            # output_pre = model(img.to(device)).cpu()
            output_pre = torch.softmax(output_pre, dim=0)
            # predict = torch.softmax(output_pre, dim=0)
        # Append model outputs to the output array
        # if sample_dir == "":
        #     a = torch.softmax(torch.squeeze(output_pre), dim=0)*100
        #     print(["{:.2f}%".format(percent) for percent in a])
        output.append(output_pre)

    # Compute weights for cranial views
    current_dir = os.path.abspath(__file__)
    weight_cache_dir = os.path.join(os.path.dirname(current_dir), "__pycache__")
    sensitivity = 10
    if "weight_cache.npy" not in os.listdir(weight_cache_dir):
        ACC_MAX = max(s_l_acc, s_d_acc, s_v_acc, m_d_acc)
        s_l_weight = 1 - (ACC_MAX - s_l_acc) * sensitivity
        s_d_weight = 1 - (ACC_MAX - s_d_acc) * sensitivity
        s_v_weight = 1 - (ACC_MAX - s_v_acc) * sensitivity
        m_d_weight = 1 - (ACC_MAX - m_d_acc) * sensitivity
        weight_list = [s_d_weight, s_l_weight, s_v_weight, m_d_weight]
        np.save(os.path.join(weight_cache_dir, "weight_cache.npy"), weight_list)
    else:
        weight_list = np.load(os.path.join(weight_cache_dir, "weight_cache.npy"))
        s_d_weight = weight_list[0]
        s_l_weight = weight_list[1]
        s_v_weight = weight_list[2]
        m_d_weight = weight_list[3]

    i = 0
    # Second pass: iterate all sample images for one individual
    weight_average_predict = torch.Tensor(weight_average_predict)
    a = weight_average_predict.shape
    # Compute weighted average
    for image in images:
        image = ut.text_segmentation(image, ".")[0]
        image_flag = image.split("#")[4] + "#" + image.split("#")[5]
        if len(image.split("#")) > 6:
            image_flag = "p"
        if image_flag == "p":
            weight_average_predict += 0.5 * output[i]
        elif image_flag == "s#l":
            weight_average_predict += s_l_weight * output[i]
        elif image_flag == "s#d":
            weight_average_predict += s_d_weight * output[i]
        elif image_flag == "s#v":
            weight_average_predict += s_v_weight * output[i]
        elif image_flag == "m#l":
            weight_average_predict += m_d_weight * output[i]
        i += 1
    # Select class with highest confidence from weighted result
    weight_average_predict = torch.argmax(weight_average_predict).numpy()

    return weight_average_predict
