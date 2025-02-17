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
from torcheeg.datasets.constants import DREAMER_CHANNEL_LOCATION_DICT, DEAP_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants import DREAMER_ADJACENCY_MATRIX, DEAP_ADJACENCY_MATRIX
from torcheeg.datasets import DREAMERDataset, DEAPDataset
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.model_selection import train_test_split_groupby_trial, train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Local Imports
from utils.checkpoint import train_and_save,  train_validate_and_save, train_validate_test_and_save, tvt_save_acc_loss_f1
from utils.log import get_logger
from utils.utils import print_var, train_one_epoch, train_one_epoch_lstm, get_num_params, train_one_step_tqdm
from utils.transforms import STFTSpectrogram, TORCHEEGBaselineCorrection

from models.STFT_Spectrogram.stft_cnn import STFT_Two_Layer_CNN_Pro, STFT_Three_Layer_CNN_Pro
from models.STFT_Spectrogram.stft_cnn_lstm import STFT_LSTM_CNN_Model
from models.cnn_based import  UNET_VIT
from models.cnn_based import UNET_VIT_TSception, UNET_VIT_INCEPTION
from models.the_model import *

if __name__ == "__main__":
    rng_num =  2024 #122
    batch_size = 32

    dataset_name = 'DEAP_STFT33_BC_After'
    emotion_dims = ['valence', 'dominance',  'arousal']

    for model_num in range(2):
        for emotion_dim in emotion_dims:
            print(f"\n{emotion_dim} , {model_num}\n")
            try: 
                del dataset, train_loader, val_loader, test_loader ,model, optimizer, loss_fn
            except:
                pass

            root_path = './raw_data/DEAP'
            io_path = f'./saves/datasets/{dataset_name}'  # IO path to store the dataset

            # Import data
            dataset = DEAPDataset(io_path=f"{io_path}",
                                    root_path=root_path,
                                    offline_transform=transforms.Compose([
                                        # transforms.BaselineRemoval(),
                                        STFTSpectrogram(n_fft=64, hop_length=4, contourf=False, apply_to_baseline=True), # [batch,14, 33, 33]
                                        TORCHEEGBaselineCorrection(),
                                    ]),
                                    online_transform=transforms.Compose([
                                        # transforms.MeanStdNormalize(apply_to_baseline=True),#MeanStdNormalize() , MinMaxNormalize()
                                        transforms.ToTensor(),
                                    ]),
                                    label_transform=transforms.Compose([
                                        transforms.Select(emotion_dim),
                                        transforms.Binary(threshold=5.0), 
                                    ]),
                                    chunk_size=128,
                                    overlap = 0, # 0 seconds
                                    num_worker=4)


            # Split train val test 
            split_type = 'group_by_trial'
            if split_type == 'group_by_trial':
                train_dataset, test_dataset = train_test_split_groupby_trial(dataset= dataset, test_size = 0.2, shuffle= True) #, random_state= rng_num)
                train_dataset, val_dataset = train_test_split_groupby_trial(dataset= train_dataset, test_size = 0.2, shuffle=True) #, random_state= rng_num)
            elif split_type == 'simple':
                train_dataset, test_dataset = train_test_split(dataset= dataset, test_size = 0.2, shuffle= True) #, random_state= rng_num)
                train_dataset, val_dataset = train_test_split(dataset= train_dataset, test_size = 0.2, shuffle=True) #, random_state= rng_num)

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
            model = None
            if model_num == 1:
                model = UNET_DGCNN_INCEPTION_GAT_Transformer_Parallel(
                    in_channels=dataset[0][0].shape[0], unet_feature_channels=[64,128,256], 
                    graph_feature_size=5, dgcnn_layers=4, dgcnn_hid_channels=32, num_heads=4, 
                    n_classes=2, dropout=0.5, bias=True, linear_hid=64)
            elif model_num == 0:
                model = DGCNN_INCEPTION_GAT_Transformer_Parallel(
                    in_channels=dataset[0][0].shape[0], 
                    graph_feature_size=5, dgcnn_layers=4, dgcnn_hid_channels=32, num_heads=4, 
                    n_classes=2, dropout=0.5, bias=True, linear_hid=64)

            assert model != None

            print(f"Selected model name : {model.__class__.__name__}")
            print_var("Model is ", model)
            print('*' * 30)
        
            # ****************** Choose your Loss Function ******************************
            # loss_fn = nn.BCEWithLogitsLoss()
            # loss_fn = nn.MSELoss()
            loss_fn = nn.CrossEntropyLoss()
        
            # ****************** Choose your Optimizer ******************************
            # optimizer = optim.Adam(model.parameters(), lr=0.01) # lr = 0.0001  0.001
            # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.937,weight_decay=1e-5) # SCHEDULE!
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.937,weight_decay=1e-5) # TRAIN!


            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            num_epochs = 40 # 300 500 600 800
            model_name = dataset_name + "_" + model.__class__.__name__  

            print(f"Start training for {num_epochs} epoch")

            model = model.to(device)
            loss_hist, acc_hist , loss_val_hist , \
            acc_val_hist, loss_test, acc_test ,\
            (f1_hist, f1_val_hist, f1_test) = tvt_save_acc_loss_f1(model, 
                                                                    dataset_name, 
                                                                    model_name, 
                                                                    emotion_dim, 
                                                                    train_loader, 
                                                                    val_loader,
                                                                    test_loader,  
                                                                    optimizer, 
                                                                    loss_fn, 
                                                                    device, 
                                                                    num_epochs=num_epochs,
                                                                    is_binary= False,
                                                                    num_classes= 2,
                                                                    en_shcheduler=False , # Enable lr scheduling
                                                                    step_size=[10], #15
                                                                    gamma=0.1
                                                                ) 

            
            print("Training process is done!")
            print(f"Test: LOSS: {loss_test}, ACC: {acc_test}, F1: {f1_test}")
            print(f"Model parameter count: {get_num_params(model,1)}")

