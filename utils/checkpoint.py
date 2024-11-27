import sys
import os

# Dynamically add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  

import time
import json
import torch
import matplotlib.pyplot as plt
from utils.utils import get_num_params, train_one_step_tqdm
from utils.log import get_logger

def load_model(model_path, model_class, optimizer=None, **kwargs):
    """
    Load a model and its state from a checkpoint.

    Parameters:
    - model_path: The file path of the saved model.
    - model_class: The model class (to recreate the model).
    - optimizer: The optimizer object to load state into (optional).
    - kwargs: Additional arguments passed to the model class for initialization.
    
    Returns:
    - model: The model loaded with its state.
    - optimizer: The optimizer (if provided and loaded).
    - epoch: The last epoch saved.
    - loss: The last loss value saved.
    - accuracy: The last accuracy value saved.
    """
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        
        # Initialize model and load the state_dict
        model = model_class(**kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize optimizer if provided and load its state_dict
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        
        print(f"Model loaded from {model_path}, epoch {epoch}, loss {loss}, accuracy {accuracy}")
        return model, optimizer, epoch, loss, accuracy
    else:
        print(f"No checkpoint found at {model_path}. Initializing a new model.")
        model = model_class(**kwargs)
        return model, optimizer, None, None, None



def create_save_directory(dataset_name, model_name, emotion_dim):
    """
    Create the directory structure for saving the model and other information.
    
    Parameters:
    - dataset_name: str, the dataset being used.
    - model_name: str, the model being used.
    - emotion_dim: str, the target variable (valence, dominance, or arousal).
    
    Returns:
    - save_path: str, the path where the model and data will be saved.
    """

    # run_file = f'./run_numbers.json'
    run_file = os.path.join('saves', 'models','run_numbers.json')
   
    # Load or initialize run numbers
    if os.path.exists(run_file):
        with open(run_file, 'r') as file:
            run_numbers = json.load(file)
    else:
        run_numbers = {}
    
    # Ensure the emotion_dim exists and initialize it as a dictionary if necessary
    if emotion_dim not in run_numbers:
        run_numbers[emotion_dim] = {}

    # Initialize the model_name if it doesn't exist for the current emotion_dim
    if model_name not in run_numbers[emotion_dim]:
        run_numbers[emotion_dim][model_name] = 0

    # Increment the run number for the current model_name and emotion_dim
    run_numbers[emotion_dim][model_name] += 1    

        
    # Save the updated run numbers back to the file
    with open(run_file, 'w') as file:
        json.dump(run_numbers, file, indent=4)


    # Get the current run number
    run_num = run_numbers[emotion_dim][model_name]
   
    # Define log file path
    log_path = os.path.join('saves', 'models', dataset_name, model_name, 'logs')
    os.makedirs(log_path, exist_ok=True)

    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")  # Get the current time
    save_path = os.path.join('saves', 'models', dataset_name, model_name, emotion_dim, str(run_num))
    os.makedirs(save_path, exist_ok=True)

    return save_path, log_path, run_num


def save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name="model_checkpoint.pth"):
    """
    Save the model checkpoint along with optimizer, epoch, loss, and accuracy.
    
    Parameters:
    - model: the model to save.
    - optimizer: the optimizer.
    - epoch: the current epoch number.
    - loss: the loss value at this epoch.
    - acc: the accuracy value at this epoch.
    - save_path: where to save the checkpoint.
    - file_name: the filename for saving the model checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': acc
    }
    
    model_save_path = os.path.join(save_path, file_name)
    torch.save(checkpoint, model_save_path)
    print(f"Model saved to {model_save_path}")


def save_training_plots(loss_hist, acc_hist, save_path, file_name_prefix="training"):
    """
    Save the loss and accuracy plots as PNG files.
    
    Parameters:
    - loss_hist: list of loss values.
    - acc_hist: list of accuracy values.
    - save_path: where to save the plots.
    - file_name_prefix: prefix for the PNG filenames.
    """
    # Save loss plot
    plt.figure()
    plt.plot(range(len(loss_hist)), loss_hist)
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    loss_plot_path = os.path.join(save_path, f"{file_name_prefix}_loss.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    # Save accuracy plot
    plt.figure()
    plt.plot(range(len(acc_hist)), acc_hist)
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    acc_plot_path = os.path.join(save_path, f"{file_name_prefix}_accuracy.png")
    plt.savefig(acc_plot_path)
    print(f"Accuracy plot saved to {acc_plot_path}")
    

# TODO: save only the last model and log the outputs in a file
# TODO: save the results in different run folders
# Training loop that calls the save functions
def train_and_save(model, dataset_name, model_name, emotion_dim, dataloader, optimizer, loss_fn, device, num_epochs=30):
    # Create the directory to save data
    save_path, log_path, run_num = create_save_directory(dataset_name, model_name, emotion_dim)
    log_handle = get_logger(os.path.join(log_path, f"report_{run_num}_{dataset_name}_{model_name}_{emotion_dim}.txt"))


    # Log Model and Trainer Info
    log_handle.info(f"Using data set [{dataset_name}] with emotion demention [{emotion_dim}]")
    log_handle.info(f"Training model [{model_name}]")
    log_handle.info(f"Using Optimizer [{optimizer.__class__.__name__}] with learning rate = {optimizer.param_groups[0]['lr']}")
    log_handle.info(f"Using Loss Function [{loss_fn.__class__.__name__}]")

    # Lists to store loss and accuracy values
    loss_hist = []
    acc_hist = []

    # Initialize the best loss variable to track the least loss
    best_loss = float('inf')  # Set to infinity initially to ensure it gets updated in the first epoch

    log_handle.info("Start Training")
    for epoch in range(num_epochs):
        # Model training (assuming train_one_step_tqdm is implemented)
        model, loss, acc = train_one_step_tqdm(model, dataloader, loss_fn, optimizer, device, epoch, True)
        
        log_handle.info(f"[Train] Epoch {epoch} - Loss: {loss}, Acc: {acc} ")

        loss_hist.append(loss)
        acc_hist.append(acc)
        
        # Save the model checkpoint only if the current loss is better (lower) than the best loss
        if loss < best_loss:
            best_loss = loss  # Update the best loss
            # save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_epoch_{epoch}.pth")
            save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint.pth")
            print(f"New best model saved with loss {loss:.4f} at epoch {epoch}")
        
        # save acc and loss plot each 50 epochs
        if epoch % 50 == 0  and epoch != 0:
            save_training_plots(loss_hist, acc_hist, save_path)



    # Save the training plots
    save_training_plots(loss_hist, acc_hist, save_path)
    print("Training complete and data saved!")



if __name__ == "__main__":
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
    from utils.log import get_logger
    from utils.utils import print_var, train_one_epoch, train_one_epoch_lstm, get_num_params, train_one_step_tqdm
    from models.cnn import Two_Layer_CNN, Two_Layer_CNN_Pro, Simplified_CNN
    from models.rnns import LSTM


    dataset_name = 'Dreamer_time_series_01'
    emotion_dim = 'valence'  # valence, dominance, or arousal
    model_name = 'LSTM'  # You can choose your model (e.g., 'LSTM', 'Two_Layer_CNN', etc.)
    
    mat_path = './raw_data/DREAMER.mat'  # path to the DREAMER.mat file
    io_path = f'./saves/datasets/{dataset_name}'  # IO path to store the dataset

    # Import data
    dataset = DREAMERDataset(io_path=f"{io_path}",
                            mat_path=mat_path,
                            offline_transform=transforms.Compose([
                                transforms.MeanStdNormalize(),
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

    # Create DataLoader    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Define the model, optimizer, and loss function
    model = LSTM(14, 128, 4, 1)  # Model definition
    model.to(device)  # Move model to the correct device
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    # Start training and save the model, plots, and other data
    train_and_save(model, dataset_name, model_name, emotion_dim, dataloader, optimizer, loss_fn, device,num_epochs=3)
