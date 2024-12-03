import sys
import os

# Dynamically add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import r2_score
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
import re

from utils.utils import AverageMeter

# TODO: implement Self Distillation !


def loss_fn_kd(outputs, labels, teacher_outputs, T=10, alpha=0.6):
    loss = F.kl_div(F.log_softmax(outputs/T, dim=1),
                    F.softmax(teacher_outputs/T, dim=1),
                    reduction= 'batchmean') * (alpha * T**2) + \
           F.cross_entropy(outputs, labels) * (1-alpha)
    return loss

# train is different but you must use a normal validation to check how the student model performs
def train_one_epoch_kd_tqdm(student, teacher,  train_loader, loss_fn_kd, optimizer, device, epoch=None, is_binary=True,num_classes=3):
    teacher.eval()
    student.train()
    loss_train = AverageMeter()

    if is_binary:
        acc_train = BinaryAccuracy().to(device)
    else:
        acc_train = Accuracy(task="multiclass",num_classes=num_classes).to(device)
    
    with tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch != None:
                tepoch.set_description(f"Epoch: {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.float()
            
            outputs = student(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            loss = loss_fn_kd(outputs.squeeze(), targets, teacher_outputs.squeeze(), T=10, alpha=0.6)

            loss.backward()
            # Weight Clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5) ## 0.5 NEW ADDITION
            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item())
            acc_train(outputs.squeeze(), targets.int())
            
            tepoch.set_postfix(loss= loss_train.avg,
                                accuracy= 100.0 * acc_train.compute().item())

    return student, loss_train.avg, acc_train.compute().item()

if __name__ =="__main__":
    # teacher = torch.load('teacher.pt')
    # teacher.eval()

    teacher = nn.Sequential(nn.Linear(10,30),nn.Linear(30,3))
    teacher.eval()

    student = nn.Sequential(nn.Linear(10,3))

    data = torch.randn(100,10)
    target = torch.randn(100,3)
    print(data.shape)
    print(target.shape)

    dataset = TensorDataset(data, target)
    train_loader = DataLoader(dataset, batch_size= 10, shuffle= True)
    print(len(train_loader))
    
    optimizer = optim.Adam(student.parameters(), lr=0.01)

    
    loss_hist = []
    acc_hist = []
    num_epochs = 10
    for epoch in range(num_epochs):
        student, loss, acc = train_one_epoch_kd_tqdm(student, teacher,  train_loader,\
                                                    loss_fn_kd, optimizer, device='cpu', \
                                                    epoch=epoch, is_binary=False)
        loss_hist.append(loss)
        acc_hist.append(acc)

    plt.figure(num= 1)
    plt.plot(range(num_epochs), loss_hist)
    plt.figure(num= 2)
    plt.plot(range(num_epochs), acc_hist)
    plt.show()
    