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
from utils.transforms import STFTSpectrogram

from models.STFT_Spectrogram.stft_cnn import STFT_Two_Layer_CNN_Pro, STFT_Three_Layer_CNN_Pro
from models.STFT_Spectrogram.stft_cnn_lstm import STFT_LSTM_CNN_Model
from models.cnn_based import  UNET_VIT
from models.cnn_based import UNET_VIT_TSception, UNET_VIT_INCEPTION
from models.the_model import *

if __name__ == "__main__":
    rng_num =  2024 #122
    batch_size = 32

    dataset_name = 'DEAP_STFT33'
    emotion_dim = 'valence'  # valence, dominance, or arousal
    
    root_path = './raw_data/DEAP'
    io_path = f'./saves/datasets/{dataset_name}'  # IO path to store the dataset

    # Import data
    dataset = DEAPDataset(io_path=f"{io_path}",
                            root_path=root_path,
                            offline_transform=transforms.Compose([
                                # transforms.BaselineRemoval(),
                                STFTSpectrogram(n_fft=64, hop_length=4, contourf=False, apply_to_baseline=True), # [batch,14, 33, 33]
                                transforms.MeanStdNormalize(apply_to_baseline=True),#MeanStdNormalize() , MinMaxNormalize()
                                # transforms.BaselineRemoval(),
                            ]),
                            online_transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select(emotion_dim),
                                transforms.Binary(threshold=5.0), 
                            ]),
                            chunk_size=128,
                            overlap = 0, # 0 seconds
                            num_worker=4)

    # print(dataset)
    # print(dataset[0])
    # print(dataset[0][0].shape)
    # print(dataset[0][1])

    # sys.exit()


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

    # ****************** Choose your Model ******************************
    # model = STFT_Two_Layer_CNN_Pro() ########## 95.5
    # model = STFT_Three_Layer_CNN_Pro()
    # model = STFT_LSTM_CNN_Model()

    # model = UNET_VIT(
    #     in_channels=dataset[0][0].shape[0], unet_out_channels=3,
    #     img_size=dataset[0][0].shape[1], patch_size=3, 
    #     n_classes=2, embed_dim=768, depth=5, n_heads=6,
    #     mlp_ratio=4., qkv_bias=True, p=0.5, attn_p=0.5
    # )

    # # UNET_VIT model 2 -> tHis is a good model !!!!!!!! !!!!!!!!!!!
    # model = UNET_VIT(
    #     in_channels=dataset[0][0].shape[0], unet_out_channels=3,
    #     img_size=dataset[0][0].shape[1], patch_size=5, 
    #     n_classes=2, embed_dim=256, depth=5, n_heads=8, # depth=5
    #     mlp_ratio=4., qkv_bias=True, p=0.5, attn_p=0.5 # mlp_ratio=4.
    # )

    # Note: Change sampling rate so that the Tsception kernels can have good kernel size 
    # samplig rate /2(4 and 8) + 1 =>  16/2+1 16/4+1 16/8+1 => 9,5,3
    # num_channel should be the size of stft as well 22
    # model = UNET_VIT_TSception(
    #     in_channels=dataset[0][0].shape[0],unet_out_channels=3,
    #     img_size=dataset[0][0].shape[1], patch_size=3, 
    #     n_classes=2, embed_dim=768, depth=5, n_heads=6,
    #     mlp_ratio=4., qkv_bias=True, p=0.5, attn_p=0.5,
    #     sampling_rate= 16, num_channels=22
    # )

    ###########################################################
    # GOOD MODEL, 16M !!!!!!!!!!!!!!!!!!!!!!!!! val:83%
    # model = UNET_VIT_INCEPTION(
    #     in_channels=dataset[0][0].shape[0], unet_out_channels=3, 
    #     img_size=dataset[0][0].shape[1], patch_size=3, n_classes=2, 
    #     embed_dim=128, depth=5, n_heads=8, mlp_ratio=4.0, qkv_bias=True,  # embed_dim=768, n_heads=6
    #     p=0.5, attn_p=0.5)

    # Good model, 6M ,val:%83
    # model = UNET_DGCNN_INCEPTION(in_channels=dataset[0][0].shape[0], unet_out_channels=3, unet_feature_channels=[64,128,256], n_classes=2)

    # model = UNET_DGCNN_INCEPTION2(in_channels=dataset[0][0].shape[0], unet_feature_channels=[64,128,256], graph_feature_size=5,dgcnn_layers=10, n_classes=2)

    model = UNET_DGCNN_INCEPTION_GAT(in_channels=dataset[0][0].shape[0], unet_feature_channels=[64,128,256], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, n_classes=2, dropout=0.5, bias=True)
   
   # Val: 85%
    model = UNET_DGCNN_INCEPTION_GAT(in_channels=dataset[0][0].shape[0], unet_feature_channels=[64,128,256], graph_feature_size=10, dgcnn_layers=4, dgcnn_hid_channels=64, n_classes=2, dropout=0.5, bias=True, linear_hid=128)


    print(f"Selected model name : {model.__class__.__name__}")
    # print(f"Model parameter count: {get_num_params(model,1)}")
    print_var("Model is ", model)
    print('*' * 30)
    
    # ****************** Choose your Loss Function ******************************
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    
    # ****************** Choose your Optimizer ******************************
    # optimizer = optim.Adam(model.parameters(), lr=0.01) # lr = 0.0001  0.001
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937,weight_decay=1e-5) # TRAIN!
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.937,weight_decay=1e-5) # SCHEDULE!


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_epochs = 200 # 300 500 600 800
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



    # model = model.to(device)
    # loss_hist, acc_hist , loss_val_hist , acc_val_hist, loss_test, acc_test = train_validate_test_and_save(model, 
    #                                                                                 dataset_name, 
    #                                                                                 model_name, 
    #                                                                                 emotion_dim, 
    #                                                                                 train_loader, 
    #                                                                                 val_loader,
    #                                                                                 test_loader,  
    #                                                                                 optimizer, 
    #                                                                                 loss_fn, 
    #                                                                                 device, 
    #                                                                                 num_epochs=num_epochs)


    # print("Training process is done!")
    # print(f"Test: LOSS: {loss_test}, ACC: {acc_test}")
    # print(f"Model parameter count: {get_num_params(model,1)}")

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