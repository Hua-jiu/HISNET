import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils import tensorboard
import time
from tools.Early_Stopping import EarlyStopping
from tools.GPU_Detecter import GPU_Detect
import os
import sys
import json

sys.path.append("./")


json_path = f"./data/more_species_labels.json"
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
json_file = open(json_path, "r")
class_indict = json.load(json_file)
genus_name_dict = {}
for genus_name in class_indict.keys():
    genus_name_dict[genus_name] = len(class_indict[genus_name])
genus_name = "Parascaptor"

log_txt = open(f"./docs/{genus_name}.txt", mode="a", encoding="utf-8")
batch_size = 16
learning_rate = 0.001
num_epochs = 125
step_size = 25
# 分类数
num_class = genus_name_dict[genus_name]
patience = 30

save_path = f"./weights/EB3_{genus_name}"
if os.path.exists(f"{save_path}") is False:
    os.makedirs(f"{save_path}")

log_path = f"./logs/EB3_{genus_name}"

earlt_stopping = EarlyStopping(save_path, patience, verbose=True)
device = torch.device(
    f"cuda:{GPU_Detect()}" if torch.cuda.is_available() else "cpu"
)
print(device, file=log_txt)

train_path = f"./data/train/{genus_name}"

test_path = f"./data/test/{genus_name}"
weights_path = f"./weights/efficientnetb3.pth"  #  迁移学习

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

train_data = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
)

test_data = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=2
)

print(train_data.class_to_idx, file=log_txt)
print(test_data.class_to_idx, file=log_txt)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("train_data_size:{}".format(train_data_size), file=log_txt)
print("test_data_size:{}".format(test_data_size), file=log_txt)

net = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, num_class)
net = net.to(device)

loss_func = torch.nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=step_size, gamma=0.8, last_epoch=-1
)

writer = tensorboard.SummaryWriter(f"{log_path}")
best_acc = 0.0
best_epoch = 0

start = time.time()
for epoch in range(num_epochs):
    net.train()
    total_train_loss = 0
    running_acc = 0
    print(
        f"======================Epoch {epoch + 1}=======================",
        file=log_txt,
    )
    print(
        "lr = {}".format(optimizer.state_dict()["param_groups"][0]["lr"]),
        file=log_txt,
    )
    for i, data in tqdm(enumerate(train_dataloader)):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = net.forward(imgs)
        loss = loss_func(outputs, targets)

        total_train_loss += loss.item() * targets.size(0)
        _, pred = torch.max(outputs, 1)
        num_correct = (pred == targets).sum()
        accuracy = (pred == targets).float().mean()
        running_acc += num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(
        "Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}".format(
            epoch + 1,
            total_train_loss / train_data_size,
            running_acc / train_data_size,
        ),
        file=log_txt,
    )
    writer.add_scalar("train_loss", total_train_loss / train_data_size, epoch)
    writer.add_scalar("train_acc", running_acc / train_data_size, epoch)
    scheduler.step()

    net.eval()
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_func(outputs, targets)
            eval_loss += loss.item() * targets.size(0)
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == targets).sum()
            eval_acc += num_correct.item()

        print(
            "Test Loss: {:.6f}, Acc: {:.6f}".format(
                eval_loss / test_data_size, eval_acc / test_data_size
            ),
            file=log_txt,
        )
        writer.add_scalar("test_loss", eval_loss / test_data_size, epoch)
        writer.add_scalar("test_acc", eval_acc / test_data_size, epoch)
        writer.add_scalar(
            "learning_rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch
        )
        print(file=log_txt)

    if eval_acc / test_data_size > best_acc:
        best_acc = eval_acc / test_data_size
        best_epoch = epoch + 1
    valid_acc = eval_acc / test_data_size
    earlt_stopping(valid_acc, net)
    if earlt_stopping.early_stop:
        print("Early Stopping", file=log_txt)
        break
end = time.time()
print(
    "Finish all epochs, best epoch: {}, best acc: {}".format(best_epoch, best_acc),
    file=log_txt,
)
print("running time:", end - start, file=log_txt)
writer.add_scalar("best_acc", best_acc)
writer.close()
log_txt.close()
torch.cuda.empty_cache()
