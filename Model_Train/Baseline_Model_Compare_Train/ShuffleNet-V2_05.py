# ===============
# Coding: UTF-8
# ===============
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils import tensorboard
import time
from tools.Early_Stopping import EarlyStopping
import os
from tools.GPU_Detecter import GPU_Detect

if __name__ == '__main__':

    batch_size = 512
    learning_rate = 1e-2
    num_epochs = 250
    step_size = 25
    num_class = 18
    patience = 30
    test_type = 'ShuffleNet-V2_05'
    if os.path.exists(f"./weights/{test_type}") is False:
        os.makedirs(f"./weights/{test_type}")

    save_path = f"./weights/{test_type}"
    earlt_stopping = EarlyStopping(save_path, patience, verbose=True)
    device = torch.device(f"cuda:{GPU_Detect()}" if torch.cuda.is_available() else "cpu")
    # load data
    train_path = "./data/train"
    test_path = "./data/test"

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

    print(train_data.class_to_idx)

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("train_data_size:{}".format(train_data_size))
    print("test_data_size:{}".format(test_data_size))

    # load net
    net = torchvision.models.shufflenet_v2_x0_5(weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    # print(net)
    net.fc = torch.nn.Linear(net.fc.in_features, num_class)
    net = net.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8, last_epoch=-1)

    # init tensorboard
    writer = tensorboard.SummaryWriter(f"./logs/{test_type}")
    best_acc = 0.0
    best_epoch = 0

    start = time.time()
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        running_acc = 0
        print("-------------------{}--------------------".format(epoch + 1))
        print("lr = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
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
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, total_train_loss / train_data_size, running_acc / train_data_size))
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

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_size
                                                          , eval_acc / test_data_size))
            writer.add_scalar("test_loss", eval_loss / test_data_size, epoch)
            writer.add_scalar("test_acc", eval_acc / test_data_size, epoch)
            writer.add_scalar("learning_rate", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            print()

        if eval_acc / test_data_size > best_acc:
            best_acc = eval_acc / test_data_size
            best_epoch = epoch + 1
        valid_acc = eval_acc / test_data_size
        earlt_stopping(valid_acc, net)
        if earlt_stopping.early_stop:
            print("Early Stopping")
            break

    end = time.time()
    print("Finish all epochs, best epoch: {}, best acc: {}".format(best_epoch, best_acc))
    print("running time:", end - start)
    writer.close()
