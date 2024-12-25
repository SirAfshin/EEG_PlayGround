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

''' << Pre-Processing Steps >>
1.1- Band Pass filter 0.3Hz - 50Hz
2.1- Eliminate eye movement with PCA
3.1- Extract DE (Differential Entropy) across five freq. bands [0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]
3.2- Followed by preprocessing with Linear Dynamic System (LDS)
4.1- Segment data using 1-second non-overlapping sliding windows 

Note: if sample rate is 128Hz then 1 second of data has 128 data points
'''

''' << Splitting Data >>
1. subject-dependent:
        we allocate each individual’s data into training, validation, and test sets in a 0.6:0.2:0.2 ratio.
2. cross-subject tasks:
        we similarly partition subjects into training, validation, and test sets following the same 0.6:0.2:0.2 ratio.
'''

''' << Experimental Tasks >>
(1) Subject-dependent: YES
(2) Cross-subject:     YES
(3) Subject-independent: NO
(4) Cross-session: NO

It is important to note that we treat data from different sessions of the same subject as if they originate from distinct subjects.
'''

''' <<  EvaluationMethodsandMetrics >>
In our benchmark, our evaluation method reports
the performance on the test set of the model that achieved
the highest F1 score on the validation set across all epochs.

We report both the mean and standard deviation of two
metrics, accuracy and F1 score.

Acc = (TP+TN)/(TP+TN+FP+FN)
F1 = (2xTP)/(2xTP + FP + FN)
'''


if __name__ == "__main__":
    batch_size = 256

    dataset_name = 'Dreamer_LibEER'# Overlap_NoBaselineRemoval
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
                                # transforms.BaselineRemoval(),
                                transforms.ToTensor(),
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select(emotion_dim),
                                transforms.Binary(threshold=2.5),   
                            ]),
                            chunk_size=128, # -1 would be all the data of each trial for a chunk
                            overlap = 0, 
                            io_mode = "lmdb",
                            baseline_chunk_size=128,
                            num_baseline=61,
                            num_worker=4)


    print(dataset)
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])

    sys.exit()


    # Split train val test ????????????
    train_dataset, test_dataset = train_test_split_groupby_trial(dataset= dataset, test_size = 0.2, shuffle= True)
    train_dataset, val_dataset = train_test_split_groupby_trial(dataset= train_dataset, test_size = 0.2, shuffle=True)
    

    # Create train/val/test dataloaders
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False)
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


    print(f"Selected model name : {model.__class__.__name__}")
    # print(f"Model parameter count: {get_num_params(model,1)}")
    print_var("Model is ", model)
    print('*' * 30)
    
    # ****************** Choose your Loss Function ******************************
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.MSELoss()
    
    # ****************** Choose your Optimizer ******************************
    optimizer = optim.Adam(model.parameters(), lr=0.001) # lr = 0.0001  0.001
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.937)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_epochs = 50 # 300 500 600
    model_name = "DREAMER_" + model.__class__.__name__ + "_LibEER" 

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
                                                                                    num_epochs=num_epochs,
                                                                                    is_binary= False,
                                                                                    num_classes= 2)


    print("Training process is done!")
    print(f"Test: LOSS: {loss_test}, ACC: {acc_test}")
    print(f"Model parameter count: {get_num_params(model,1)}")
