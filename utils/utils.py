import torch
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

def print_var(name, value):
    print(f"{name} : {value}")

def get_num_params(model, k=1e6):
    """
    Calculate the total number of parameters in a given machine learning model and return the 
    result in units of millions (if `k` is set to 1e6 by default).
    k=1e6 -> Million
    k=1e3 -> Killo
    k=1 -> The actual number
    Args:
        model (torch.nn.Module): The model whose parameters are to be counted. This should be 
                                 a PyTorch neural network model or any model that has parameters 
                                 accessible via the `parameters()` method.
        k (float, optional): A scaling factor (default is 1e6), which divides the total number of 
                              parameters. This is useful for expressing the number of parameters 
                              in units of millions, or any other unit based on the value of `k`.

    Returns:
        float: The total number of parameters in the model, scaled by the factor `k`. 
               If `k` is set to 1e6, the result will be in millions of parameters.
    """
    nums = sum(p.numel() for p in model.parameters()) / k
    return nums

def get_num_trainable_params(model, k=1e6):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / k
    return nums


# Helper class to track the average of a given metric (Your custom AverageMeter)
class AverageMeter:
    """Computes and stores average and current value
    Ex:
        train_loss_meter = AverageMeter()
        # Then inside train loop
        train_loss_meter.update(loss.item())
        # After for print purpose
        print(train_loss_meter.avg)
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    '''
    Early stopping to stop training when the validation loss stops improving.
    
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 5.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.01.
        verbose (bool): If True, prints a message for each improvement or stopping event. Default is False.
        save_best_model (bool): If True, saves the model with the best validation loss. Default is False.
        model (torch.nn.Module): The model to be saved. Default is None.
        save_path (str): Path where the best model will be saved if `save_best_model=True`. Default is None.
    Example:
        early_stopping = EarlyStopping(patience=5, delta=0.01)

        for epoch in range(num_epochs):
            # Your training and validation code...
            if early_stopping(loss_val):
                print("Early stopping triggered!")
                break
    '''
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_step_tqdm(model, train_loader, loss_fn, optimizer, device, epoch=None, is_binary=True ,num_classes=None):
    model.train()
    loss_train = AverageMeter()

    if is_binary:
        acc_train = BinaryAccuracy().to(device)
    else:
        acc_train = Accuracy(task='multiclass', num_classes= num_classes).to(device)
    
    with tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch != None:
                tepoch.set_description(f"Epoch: {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)

            # remove nan data 
            inputs = torch.stack([data for data in inputs if ~torch.isnan(data).any()])
            targets = torch.stack([targets[i] for (i,data) in enumerate(inputs) if ~torch.isnan(data).any()])

            if is_binary:
                targets = targets.float()
            
            outputs = model(inputs) + 1e-8  
            loss = loss_fn(outputs.squeeze(), targets) + 1e-8  

            loss.backward()
            
            # Weight Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) ## 0.5 NEW ADDITION

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item())
            acc_train(outputs.squeeze(), targets.int())
            
            tepoch.set_postfix(loss= loss_train.avg,
                                accuracy= 100.0 * acc_train.compute().item())

    return model, loss_train.avg, acc_train.compute().item()


def validation(model, test_loader, loss_fn, device='cpu', is_binary=True, num_classes=None):
    model.eval()
    with torch.no_grad():
        loss_valid = AverageMeter()
        if is_binary:
            acc_valid = BinaryAccuracy().to(device)
        else:
            acc_valid = Accuracy(task='multiclass', num_classes= num_classes).to(device)

        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # remove nan data 
            inputs = torch.stack([data for data in inputs if ~torch.isnan(data).any()])
            targets = torch.stack([targets[i] for (i,data) in enumerate(inputs) if ~torch.isnan(data).any()])

            if is_binary:
                targets = targets.float()

            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), targets)

        loss_valid.update(loss.item())
        acc_valid(outputs.squeeze(), targets.int())
    return loss_valid.avg, acc_valid.compute().item()

def validation_with_tqdm(model, test_loader, loss_fn, device='cpu', is_binary=True, num_classes=None):
    model.eval()
    with torch.no_grad():
        loss_valid = AverageMeter()
        if is_binary:
            acc_valid = BinaryAccuracy().to(device)
        else:
            acc_valid = Accuracy(task='multiclass', num_classes= num_classes).to(device)


    with tqdm(test_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            tepoch.set_description(f"Validation - ")

            inputs = inputs.to(device)
            targets = targets.to(device)

            # remove nan data 
            inputs = torch.stack([data for data in inputs if ~torch.isnan(data).any()])
            targets = torch.stack([targets[i] for (i,data) in enumerate(inputs) if ~torch.isnan(data).any()])

            if is_binary:
                targets = targets.float()

            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), targets)

            loss_valid.update(loss.item())
            acc_valid(outputs.squeeze(), targets.int())
            tepoch.set_postfix(loss= loss_valid.avg,
                                accuracy= 100.0 * acc_valid.compute().item())

    return loss_valid.avg, acc_valid.compute().item()


def train_one_epoch(model, optimizer, loss_fn, data_loader, device, epoch= None):
    model = model.to(device)
    model.train()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.unsqueeze(1)
        targets = targets.float()
        predictions = model(inputs)
        # print(predictions)
        loss = loss_fn(targets, predictions.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Report data section
    if epoch != None:
        print(f"Epoch {epoch}: Train loss= {loss.item()}")
    else:
        print(f"Train loss= {loss.item()}")
    
    return model, loss.item()


def train_one_epoch_lstm(model, optimizer, loss_fn, data_loader, device, epoch= None):
    model = model.to(device)
    model.train()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.permute(0, 2, 1) # (Batch, Length, features = 14 channels)

        # inputs = inputs.unsqueeze(1)
        targets = targets.float()
        predictions = model(inputs)
        # print(predictions)
        loss = loss_fn(targets, predictions.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Report data section
    if epoch != None:
        print(f"Epoch {epoch}: Train loss= {loss.item()}")
    else:
        print(f"Train loss= {loss.item()}")
    
    return model, loss.item()


def get_loss_acc_from_log(file_path):
    # Regular expression pattern to match loss and accuracy values
    pattern = r"Loss:\s([0-9\.]+),\sAcc:\s([0-9\.]+)"

    # Initialize lists to store loss and accuracy values
    losses = []
    accuracies = []

    # Open the file and extract the desired values
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Search for the pattern in the line
            match = re.search(pattern, line)
            if match:
                # Extract the loss and accuracy values and append to lists
                loss = float(match.group(1))
                acc = float(match.group(2))
                losses.append(loss)
                accuracies.append(acc)
    return losses, accuracies



if __name__ == "__main__":
    seed = 100
    # torch.random.manual_seed(seed)  
    data = torch.randn(100,2)
    target = torch.randn(100,1)
    print(data.shape)
    print(target.shape)
    dataset = TensorDataset(data, target)
    dataloader = DataLoader(dataset, batch_size= 10, shuffle= True)

    print(len(dataloader))

    model = nn.Sequential(nn.Linear(2,10),
                            nn.ReLU(),
                            nn.Linear(10,1))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_hist = []
    num_epochs = 10
    for epoch in range(num_epochs):
        model, loss = train_one_epoch(model, optimizer, loss_fn, dataloader, 'cpu', epoch)
        loss_hist.append(loss)

    plt.plot(range(num_epochs), loss_hist)
    plt.show()
    
