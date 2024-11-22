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


def print_var(name, value):
    print(f"{name} : {value}")

def get_num_params(model, k=1e6):
    """
    Calculate the total number of parameters in a given machine learning model and return the 
    result in units of millions (if `k` is set to 1e6 by default).
    k=1e6 -> Million
    k=1e3 -> Killo
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

def train_one_step_tqdm(model, train_loader, loss_fn, optimizer, device, epoch=None, is_binary=True):
    model.train()
    loss_train = AverageMeter()

    if is_binary:
        acc_train = BinaryAccuracy().to(device)
    else:
        acc_train = Accuracy().to(device)
    
    with tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch != None:
                tepoch.set_description(f"Epoch: {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.float()
            
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item())
            acc_train(outputs.squeeze(), targets.int())
            
            tepoch.set_postfix(loss= loss_train.avg,
                                accuracy= 100.0 * acc_train.compute().item())

    return model, loss_train.avg, acc_train.compute().item()


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

def train_one_epoch_tqdm():
    pass




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
    
