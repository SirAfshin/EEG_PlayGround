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
from torch.utils.data import DataLoader , ConcatDataset
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
from utils.transforms import *

from models.STFT_Spectrogram.stft_cnn import STFT_Two_Layer_CNN_Pro
from models.STFT_Spectrogram.stft_cnn_lstm import STFT_LSTM_CNN_Model


_DataSets = ['Dreamer_time_series_01',
             'Dreamer_Freq_01',
             'Dreamer_Freq_Bandfeatures_01',
             'Dreamer_STFT_Spectrogram'
             ]


augmentation_transforms = transforms.Compose([
    TimeShiftEEG(max_shift=10),
    TimeStretchEEG(stretch_factor=1.1),
    GaussianNoiseEEG(noise_factor=0.02),
    FrequencyMaskingEEG(freq_mask_param=0.2),
    ChannelDropoutEEG(dropout_prob=0.2),
])

if __name__ == "__main__":
    rng_num =  2024 #122
    batch_size = 32

    dataset_name = 'Dreamer_STFT_Spectrogram'
    emotion_dim = 'valence'  # valence, dominance, or arousal

    mat_path = './raw_data/DREAMER.mat'  # path to the DREAMER.mat file
    io_path = f'./saves/datasets/{dataset_name}'  # IO path to store the dataset

    # Import mian data
    dataset_STFT = DREAMERDataset(io_path=f"{io_path}",
                            mat_path=mat_path,
                            offline_transform=transforms.Compose([
                                STFTSpectrogram(n_fft=64, hop_length=32, contourf=False),
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


    # Import Augmented data
    dataset_name = 'Dreamer_STFT_Augmented'
    io_path = f'./saves/datasets/{dataset_name}'  # IO path to store the dataset
    dataset_STFT_Augmented = DREAMERDataset(io_path=f"{io_path}",
                            mat_path=mat_path,
                            offline_transform=transforms.Compose([
                                augmentation_transforms,  # augmentations
                                STFTSpectrogram(n_fft=64, hop_length=32, contourf=False),
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


    # print(dataset_STFT_Augmented)
    # print(dataset_STFT_Augmented[0])
    # print(dataset_STFT_Augmented[0][0].shape)
    # print(dataset_STFT_Augmented[0][1])

    # sys.exit()


    # Split train val test 
    train_dataset, test_dataset = train_test_split_groupby_trial(dataset= dataset_STFT, test_size = 0.2, shuffle= True, random_state= rng_num)
    train_dataset, val_dataset = train_test_split_groupby_trial(dataset= train_dataset, test_size = 0.2, shuffle=True, random_state= rng_num)
    # Add augmented data to train dataset
    train_dataset = ConcatDataset([train_dataset, dataset_STFT_Augmented]) 

    # Create train/val/test dataloaders
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

    print("Dataset is ready!")
    print(f"Dataset size: {len(dataset_STFT_Augmented)}")
    print(f"Train Size: {len(train_dataset)}, Validation Size: {len(val_dataset)} , Test Size: {len(test_dataset)}")
    print(f"Input data shape: {dataset_STFT_Augmented[0][0].shape}")
    print(f"Output data (one sample): {dataset_STFT_Augmented[0][1]}")

    print('*' * 30)
    
    print_var("Number of batches inside train dataloader",len(train_loader))
    print_var("Number of batches inside validation dataloader",len(val_loader))
    print_var("Number of batches inside test dataloader",len(test_loader))

    print('*' * 30)

    # ****************** Choose your Model ******************************
    # model = STFT_Two_Layer_CNN_Pro() ########## 95.5
    model = STFT_LSTM_CNN_Model()

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
    model_name = model.__class__.__name__

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