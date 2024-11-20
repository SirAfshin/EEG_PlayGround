import sys
import os

# Dynamically add the root directory to sys.path
# Assumes that 'models' and 'utils' are in the same project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

import math
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheeg import transforms
from torch.utils.data import DataLoader
from torcheeg.datasets.constants import DREAMER_CHANNEL_LOCATION_DICT
from torcheeg.datasets import DREAMERDataset
from torcheeg.model_selection import KFoldGroupbyTrial
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Local Imports
from utils.utils import print_var, train_one_epoch, train_one_epoch_lstm
from models.cnn import Two_Layer_CNN, Two_Layer_CNN_Pro, Simplified_CNN
from models.rnns import LSTM




if __name__ == "__main__":
    emotion_dim = 'valence'  # valence, dominance, or arousal
    mat_path = './raw_data/DREAMER.mat'  # path to the DREAMER.mat file
    io_path = './saves/datasets/Dreamer_time_series_01'  # IO path to store the dataset

    # Import data
    dataset = DREAMERDataset(io_path=f"{io_path}",
                            mat_path=mat_path,
                            offline_transform=transforms.Compose([
                                transforms.MeanStdNormalize(),#MeanStdNormalize() , MinMaxNormalize()
                            ]),
                            online_transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select(emotion_dim),
                                transforms.Binary(threshold=2.5), 
                            ]),
                            chunk_size=128,
                            baseline_chunk_size=128,
                            num_baseline=61,
                            num_worker=4)



    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])

    print('Making DataLoader')
    dataloader = DataLoader(dataset, batch_size= 128, shuffle= True)
    print_var("len(dataloader)",len(dataloader))

    # model = Two_Layer_CNN()
    # model = Two_Layer_CNN_Pro()
    # model = Simplified_CNN()
    # model = LSTM(128,64,2,1) # IT should be L*F
    model = LSTM(14,64,2,1) # Should take 14 input features not 128 of the length 

    
    print_var("Model is ", model)

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loss_hist = []
    num_epochs = 30
    print("\nStart Training Process")
    for epoch in range(num_epochs):
        # model, loss = train_one_epoch(model, optimizer, loss_fn, dataloader, device, epoch)
        model, loss = train_one_epoch_lstm(model, optimizer, loss_fn, dataloader, device, epoch)

        loss_hist.append(loss)
    print("Done!")
    plt.plot(range(num_epochs), loss_hist)
    plt.show()
    
