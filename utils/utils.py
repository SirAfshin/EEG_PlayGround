import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def print_var(name, value):
    print(f"{name} : {value}")

def AverageMeter():
    def __init__():
        pass


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
    
