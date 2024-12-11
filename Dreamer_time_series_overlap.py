# More freq bins and time domain features [ batch , 14 , 65 , 65]

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
from torcheeg.datasets.constants import DREAMER_ADJACENCY_MATRIX
from torcheeg.datasets import DREAMERDataset
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.model_selection import train_test_split_groupby_trial

from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Local Imports
from utils.checkpoint import train_and_save,  train_validate_and_save, train_validate_test_and_save
from utils.log import get_logger
from utils.utils import print_var, train_one_epoch, train_one_epoch_lstm, get_num_params, train_one_step_tqdm
from utils.transforms import STFTSpectrogram

from models.cnn import Two_Layer_CNN, Two_Layer_CNN_Pro, Simplified_CNN
from models.rnns import LSTM
from models.cnn_lstm import LSTM_CNN_Model
from models.Tsception import TSCEPTIONModel
from models.YoloV9 import YOLO9_Backbone_Classifier
from models.eegnet import EEGNet_Normal_data
from models.Transformer import VanillaTransformer_time


_DataSets = ['Dreamer_time_series_01',
             'Dreamer_Freq_01',
             'Dreamer_Freq_Bandfeatures_01',
             'Dreamer_STFT_Spectrogram',
             'Dreamer_STFT_Spectrogram_02',
             'Dreamer_STFT_Spectrogram_17x17',
             'Dreamer_time_series_overlap',
             ]

if __name__ == "__main__":
    rng_num =  2024 #122
    batch_size = 256

    dataset_name = 'Dreamer_time_series_overlap'
    emotion_dim = 'valence'  # valence, dominance, or arousal
    
    mat_path = './raw_data/DREAMER.mat'  # path to the DREAMER.mat file
    io_path = f'./saves/datasets/{dataset_name}'  # IO path to store the dataset

    # Import data
    dataset = DREAMERDataset(io_path=f"{io_path}",
                            mat_path=mat_path,
                            offline_transform=transforms.Compose([
                                # normalize along the second dimension (temproal dimension)
                                transforms.MeanStdNormalize(axis=1, apply_to_baseline=True),# MeanStdNormalize() , MinMaxNormalize()
                            ]),
                            online_transform=transforms.Compose([
                                transforms.BaselineRemoval(),
                                transforms.ToTensor(),
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select(emotion_dim),
                                transforms.Binary(threshold=2.5),   
                            ]),
                            chunk_size=128, # -1 would be all the data of each trial for a chunk
                            overlap = 64, # Half of each data overlaps with the next one
                            io_mode = "lmdb",
                            baseline_chunk_size=128,
                            num_baseline=61,
                            num_worker=4)


    # print(dataset)
    # print(dataset[0])
    # print(dataset[0][0].shape)
    # print(dataset[0][1])

    # sys.exit()


    # Split train val test 
    train_dataset, test_dataset = train_test_split_groupby_trial(dataset= dataset, test_size = 0.2, shuffle= True, random_state= rng_num)
    train_dataset, val_dataset = train_test_split_groupby_trial(dataset= train_dataset, test_size = 0.1, shuffle=True, random_state= rng_num)
    

    # Create train/val/test dataloaders
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

    print("Dataset is ready!")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train Size: {len(train_dataset)}, Validation Size: {len(val_dataset)} , Test Size: {len(test_dataset)}")
    print(f"Input data shape: {dataset[0][0].shape}")
    print(f"Output data (one sample): {dataset[0][1]}")

    print('*' * 30)
    
    print_var("Number of batches inside train dataloader",len(train_loader))
    print_var("Number of batches inside validation dataloader",len(val_loader))
    print_var("Number of batches inside test dataloader",len(test_loader))

    print('*' * 30)

    # ****************** Choose your Model ******************************
    # model = Two_Layer_CNN()
    # model = Two_Layer_CNN_Pro() ####################w 74.5
    # model = Simplified_CNN()
    # model = LSTM(128,64,2,1) # IT should be L*F
    # model = LSTM(14,256,4,1) # Should take 14 input features not 128 of the length  ##############w 
    # model = LSTM_CNN_Model() ########## 95.5
    # model = TSCEPTIONModel()  ############ 
    # model = YOLO9_Backbone_Classifier()
    # model = EEGNet_Normal_data()
    # model = TSCEPTIONModel() #### validation is Ok almost
    model = VanillaTransformer_time()

    print(f"Selected model name : {model.__class__.__name__}")
    # print(f"Model parameter count: {get_num_params(model,1)}")
    print_var("Model is ", model)
    print('*' * 30)
    
    # ****************** Choose your Loss Function ******************************
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.MSELoss()
    
    # ****************** Choose your Optimizer ******************************
    optimizer = optim.Adam(model.parameters(), lr=0.001) # lr = 0.0001  0.001
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.937)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_epochs = 20 # 300 500 600
    model_name = model.__class__.__name__ + "_overlap" 

    print(f"Start training for {num_epochs} epoch")

    model = model.to(device)
    loss_hist, acc_hist , loss_val_hist , acc_val_hist, loss_test, acc_test = train_validate_test_and_save(model, 
                                                                                    dataset_name, 
                                                                                    model_name, 
                                                                                    emotion_dim, 
                                                                                    train_loader, 
                                                                                    val_loader,
                                                                                    test_loader,  
                                                                                    optimizer, 
                                                                                    loss_fn, 
                                                                                    device, 
                                                                                    num_epochs=num_epochs)


    print("Training process is done!")
    print(f"Test: LOSS: {loss_test}, ACC: {acc_test}")
    print(f"Model parameter count: {get_num_params(model,1)}")

    # # Plot Losses
    # plt.figure()
    # plt.plot(range(len(loss_hist)), loss_hist)
    # plt.plot(range(len(loss_val_hist)), loss_val_hist)
    # plt.legend(["Train Loss", "Val Loss"], loc="lower right")
    # plt.title('Loss over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    # # plot Accuracies
    # plt.figure()
    # plt.plot(range(len(acc_hist)), acc_hist)
    # plt.plot(range(len(acc_val_hist)), acc_val_hist)
    # plt.legend(["Train Acc", "Val Acc"], loc="lower right")
    # plt.title('Acc over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Acc')
    # plt.show()
    

# transforms.Concatenate([
#     transforms.BandDifferentialEntropy(),
#     transforms.BandMeanAbsoluteDeviation()])