import sys
import os

# Dynamically add the root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  

import time
import json
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import  f1_score

from utils.utils import get_num_params, train_one_step_tqdm , validation_with_tqdm
from utils.utils import train_one_step_tqdm_withF1, validation_with_tqdm_withF1
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



def create_save_directory(dataset_name, model_name, emotion_dim, pre_path= '.'):
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
    run_file = os.path.join(pre_path, 'saves', 'models','run_numbers.json')
   
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
   
    # # Define log file path
    # log_path = os.path.join('saves', 'models', dataset_name, model_name, 'logs')
    # os.makedirs(log_path, exist_ok=True)
    # Define log file path  for specific emotion dimension
    log_emotion_path = os.path.join(pre_path, 'saves', 'models', dataset_name, model_name, 'logs', emotion_dim)
    if not os.path.exists(log_emotion_path):
        os.makedirs(log_emotion_path)

    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")  # Get the current time
    save_path = os.path.join(pre_path, 'saves', 'models', dataset_name, model_name, emotion_dim, str(run_num))
    os.makedirs(save_path, exist_ok=True)

    # return save_path, log_path, run_num
    return save_path, log_emotion_path, run_num


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


def save_training_plots(loss_hist, acc_hist, save_path, file_name_prefix="training", title1="Loss", title2="Accuracy"):
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
    plt.title(f'{title1} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(title1)
    loss_plot_path = os.path.join(save_path, f"{file_name_prefix}_{title1.lower()}.png")
    plt.savefig(loss_plot_path)
    print(f"{title1} plot saved to {loss_plot_path}")
    
    # Save accuracy plot
    plt.figure()
    plt.plot(range(len(acc_hist)), acc_hist)
    plt.title(f'{title2} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(title2)
    acc_plot_path = os.path.join(save_path, f"{file_name_prefix}_{title2.lower()}.png")
    plt.savefig(acc_plot_path)
    print(f"{title2} plot saved to {acc_plot_path}")
    

# Training loop that calls the save functions
def train_and_save(model, dataset_name, model_name, emotion_dim, dataloader, optimizer, loss_fn, device, num_epochs=30):
    # Create the directory to save data
    save_path, log_path, run_num = create_save_directory(dataset_name, model_name, emotion_dim)
    log_handle = get_logger(os.path.join(log_path, f"report_{run_num}_{dataset_name}_{model_name}_{emotion_dim}.txt"))


    # Log Model and Trainer Info
    log_handle.info("Using only train save function!")
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


def train_validate_and_save(model, dataset_name, model_name, emotion_dim, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=30):
    # Create the directory to save data
    save_path, log_path, run_num = create_save_directory(dataset_name, model_name, emotion_dim)
    log_handle = get_logger(os.path.join(log_path, f"report_{run_num}_{dataset_name}_{model_name}_{emotion_dim}.txt"))
    

    # Log Model and Trainer Info
    log_handle.info("Using train and validate save function!")
    log_handle.info(f"Using data set [{dataset_name}] with emotion demention [{emotion_dim}]")
    log_handle.info(f"Training model [{model_name}]")
    log_handle.info(f"Using Optimizer [{optimizer.__class__.__name__}] with learning rate = {optimizer.param_groups[0]['lr']}")
    log_handle.info(f"Using Loss Function [{loss_fn.__class__.__name__}]")

    # Lists to store loss and accuracy values
    loss_hist = []
    acc_hist = []
    loss_val_hist = []
    acc_val_hist = []

    # Initialize the best loss variable to track the least loss
    best_loss = float('inf')  # Set to infinity initially to ensure it gets updated in the first epoch

    log_handle.info("Start Training")
    for epoch in range(num_epochs):
        # Model training (assuming train_one_step_tqdm is implemented)
        model, loss, acc = train_one_step_tqdm(model, train_loader, loss_fn, optimizer, device, epoch, True)
        # Model validating (using validation loader)
        loss_val, acc_val = validation_with_tqdm(model,val_loader, loss_fn, device, is_binary=True)

        log_handle.info(f"[Train] Epoch {epoch} - Loss: {loss}, Acc: {acc} ")
        log_handle.info(f"[Valdiation] Epoch {epoch} - Loss_Val: {loss_val}, Acc_Val: {acc_val}")

        loss_hist.append(loss)
        acc_hist.append(acc)
        loss_val_hist.append(loss_val)
        acc_val_hist.append(acc_val)
        
        # Save the model checkpoint only if the current loss is better (lower) than the best loss
        if loss < best_loss:
            best_loss = loss  # Update the best loss
            # save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_epoch_{epoch}.pth")
            save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint.pth")
            print(f"New best model saved with loss {loss:.4f} at epoch {epoch}")
        
        # save acc and loss plot each 50 epochs
        if epoch % 50 == 0  and epoch != 0:
            save_training_plots(loss_hist, acc_hist, save_path)
            save_training_plots(loss_val_hist, acc_val_hist, save_path,file_name_prefix="validation")



    # Save the training plots
    save_training_plots(loss_hist, acc_hist, save_path)
    save_training_plots(loss_val_hist, acc_val_hist, save_path,file_name_prefix="validation")
    log_handle.info(f"Model Parameter Count: {get_num_params(model,1)} ")
    
    print("Training complete and data saved!")

    return loss_hist, acc_hist , loss_val_hist , acc_val_hist


def train_validate_test_and_save(model, dataset_name, model_name, emotion_dim, train_loader, val_loader, test_loader, optimizer, loss_fn, device, num_epochs=30, is_binary= True, num_classes= None):
    # Create the directory to save data
    save_path, log_path, run_num = create_save_directory(dataset_name, model_name, emotion_dim)
    print(log_path)
    log_handle = get_logger(os.path.join(log_path, f"report_{run_num}_{dataset_name}_{model_name}_{emotion_dim}.txt"))
    

    # Log Model and Trainer Info
    log_handle.info("Using train and validate save function!")
    log_handle.info(f"Using data set [{dataset_name}] with emotion demention [{emotion_dim}]")
    log_handle.info(f"Training model [{model_name}]")
    log_handle.info(f"Using Optimizer [{optimizer.__class__.__name__}] with learning rate = {optimizer.param_groups[0]['lr']}")
    log_handle.info(f"Using Loss Function [{loss_fn.__class__.__name__}]")

    # Lists to store loss and accuracy values
    loss_hist = []
    acc_hist = []
    loss_val_hist = []
    acc_val_hist = []

    # Initialize the best loss variable to track the least loss
    best_loss = float('inf')  # Set to infinity initially to ensure it gets updated in the first epoch
    best_acc = -1.0 * float('inf')

    log_handle.info("Start Training")
    for epoch in range(num_epochs):
        # Model training (assuming train_one_step_tqdm is implemented)
        model, loss, acc = train_one_step_tqdm(model, train_loader, loss_fn, optimizer, device, epoch, is_binary=is_binary, num_classes=num_classes)
        # Model validating (using validation loader)
        loss_val, acc_val = validation_with_tqdm(model,val_loader, loss_fn, device, is_binary, num_classes)

        log_handle.info(f"[Train] Epoch {epoch} - Loss: {loss}, Acc: {acc} ")
        log_handle.info(f"[Valdiation] Epoch {epoch} - Loss_Val: {loss_val}, Acc_Val: {acc_val}")

        loss_hist.append(loss)
        acc_hist.append(acc)
        loss_val_hist.append(loss_val)
        acc_val_hist.append(acc_val)
        
        # Save the model checkpoint only if the current loss is better (lower) than the best loss
        if loss < best_loss:
            best_loss = loss  # Update the best loss
            # save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_epoch_{epoch}.pth")
            save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_loss.pth")
            print(f"New best model saved with loss {loss:.4f} at epoch {epoch}")

        # Save model if loss has not imporved but accuracy has gotten better
        if acc_val > best_acc:
            best_acc = acc_val
            save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_acc.pth")
            print(f"New best model saved with Acc {acc_val:.4f} at epoch {epoch}")

        
        # save acc and loss plot each 50 epochs
        if epoch % 50 == 0  and epoch != 0:
            save_training_plots(loss_hist, acc_hist, save_path)
            save_training_plots(loss_val_hist, acc_val_hist, save_path,file_name_prefix="validation")

    # Test model performance on test data
    loss_test, acc_test = validation_with_tqdm(model,test_loader, loss_fn, device, is_binary)
    log_handle.info(f"[Test] Loss: {loss_test} , Accuracy: {acc_test}")

    # Save the training plots
    save_training_plots(loss_hist, acc_hist, save_path)
    save_training_plots(loss_val_hist, acc_val_hist, save_path,file_name_prefix="validation")
    
    log_handle.info(f"[BEST ACC] Train: {max(acc_hist)} , Validation: {max(acc_val_hist)}")
    log_handle.info(f"Model Parameter Count: {get_num_params(model,1)} ")
    
    print(f"[BEST ACC] Train: {max(acc_hist)} , Validation: {max(acc_val_hist)}")
    print(f"Model Parameter Count: {get_num_params(model,1)} ")
    print("Training complete and data saved!")

    return loss_hist, acc_hist , loss_val_hist , acc_val_hist , loss_test, acc_test


def train_validate_test_lrschedule_and_save_(model, dataset_name, model_name, emotion_dim, train_loader, val_loader, test_loader, optimizer, loss_fn, device, num_epochs=30, is_binary=True):
    save_path, log_path, run_num = create_save_directory(dataset_name, model_name, emotion_dim)
    log_handle = get_logger(os.path.join(log_path, f"report_{run_num}_{dataset_name}_{model_name}_{emotion_dim}.txt"))
    
    # Initialize the learning rate scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.6, verbose=True)
    # early_stopping = EarlyStopping(patience=5, delta=0.01)

    # Log model and training information
    log_handle.info(f"Using dataset [{dataset_name}] with emotion dim [{emotion_dim}]")
    log_handle.info(f"Training model [{model_name}] with optimizer [{optimizer.__class__.__name__}] and learning rate = {optimizer.param_groups[0]['lr']}")
    log_handle.info("Train With Scheduler!")

    # Lists to store training and validation loss and accuracy
    loss_hist, acc_hist, loss_val_hist, acc_val_hist = [], [], [], []
    best_loss = float('inf')  # Start with an initially high loss
    
    log_handle.info("Start Training")
    for epoch in range(num_epochs):
        # Training phase
        model, loss, acc = train_one_step_tqdm(model, train_loader, loss_fn, optimizer, device, epoch, True)
        
        # Validation phase
        loss_val, acc_val = validation_with_tqdm(model, val_loader, loss_fn, device, is_binary)
        
        log_handle.info(f"[Train] Epoch {epoch} - Loss: {loss}, Acc: {acc}")
        log_handle.info(f"[Validation] Epoch {epoch} - Loss_Val: {loss_val}, Acc_Val: {acc_val}")

        # Store loss and accuracy for plotting
        loss_hist.append(loss)
        acc_hist.append(acc)
        loss_val_hist.append(loss_val)
        acc_val_hist.append(acc_val)
        
        # # Check for early stopping and save the best model
        # if early_stopping(loss_val):
        #     print("Early stopping triggered!")
        #     break
        
        # Save the model checkpoint if it's the best loss so far
        if loss_val < best_loss:
            best_loss = loss_val
            save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint.pth")
            print(f"New best model saved with validation loss {loss_val:.4f} at epoch {epoch}")
        
        # Update the learning rate scheduler
        scheduler.step(loss_val)

        # Save training/validation plots periodically
        if epoch % 50 == 0 and epoch != 0:
            save_training_plots(loss_hist, acc_hist, save_path)
            save_training_plots(loss_val_hist, acc_val_hist, save_path, file_name_prefix="validation")

    # Final test phase
    loss_test, acc_test = validation_with_tqdm(model, test_loader, loss_fn, device, is_binary)
    log_handle.info(f"[Test] Loss: {loss_test} , Accuracy: {acc_test}")

    # Save the final training/validation plots
    save_training_plots(loss_hist, acc_hist, save_path)
    save_training_plots(loss_val_hist, acc_val_hist, save_path, file_name_prefix="validation")
    
    log_handle.info(f"Model Parameter Count: {get_num_params(model, 1)}")
    print("Training complete and data saved!")

    return loss_hist, acc_hist, loss_val_hist, acc_val_hist, loss_test, acc_test


# TODO: Add torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# TODO: Also make it so that there is a parameter that lets scheduling to happen
# tvt = train validate test
def tvt_save_acc_loss_f1(model, dataset_name, model_name, emotion_dim, train_loader, val_loader, test_loader, optimizer, loss_fn, device, num_epochs=30, is_binary= True, num_classes= None, pre_path='.', en_shcheduler=False):
    # Create the directory to save data
    save_path, log_path, run_num = create_save_directory(dataset_name, model_name, emotion_dim,pre_path)
    print(log_path)
    log_handle = get_logger(os.path.join(log_path, f"report_{run_num}_{dataset_name}_{model_name}_{emotion_dim}.txt"))
    

    # Log Model and Trainer Info
    log_handle.info("Using train and validate save function!")
    log_handle.info(f"Using data set [{dataset_name}] with emotion demention [{emotion_dim}]")
    log_handle.info(f"Training model [{model_name}]")
    log_handle.info(f"Using Optimizer [{optimizer.__class__.__name__}] with learning rate = {optimizer.param_groups[0]['lr']}")
    log_handle.info(f"Using Loss Function [{loss_fn.__class__.__name__}]")
    try:
        log_handle.info(f"Model Parameter Count: {get_num_params(model,1)} ")
        print(f"Model Parameter Count: {get_num_params(model,1)} ")
    except:
        pass

    # Initialize scheduler
    scheduler = None
    if en_shcheduler == True:
        step_size = 20
        gamma = 0.1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        log_handle.info(f"Train model Using Scheduler with step of {step_size} and gamma of {gamma}")
        print(f"Train model Using Scheduler with step of {step_size} and gamma of {gamma}")

    # Lists to store loss and accuracy values
    loss_hist = []
    acc_hist = []
    loss_val_hist = []
    acc_val_hist = []
    f1_hist = []
    f1_val_hist = []

    # Initialize the best loss variable to track the least loss
    best_loss = float('inf')  # Set to infinity initially to ensure it gets updated in the first epoch
    best_acc = -1.0 * float('inf')

    log_handle.info("Start Training")
    for epoch in range(num_epochs):
        # Model training (assuming train_one_step_tqdm is implemented)
        model, loss, acc, f1 = train_one_step_tqdm_withF1(model, train_loader, loss_fn, optimizer, device, epoch, is_binary=is_binary, num_classes=num_classes)
        # Model validating (using validation loader)
        loss_val, acc_val, f1_val = validation_with_tqdm_withF1(model,val_loader, loss_fn, device, is_binary, num_classes)

        log_handle.info(f"[Train] Epoch {epoch} - Loss: {loss}, Acc: {acc}, F1: {f1} ")
        log_handle.info(f"[Valdiation] Epoch {epoch} - Loss_Val: {loss_val}, Acc_Val: {acc_val}, F1_Val: {f1_val}")

        loss_hist.append(loss)
        acc_hist.append(acc)
        loss_val_hist.append(loss_val)
        acc_val_hist.append(acc_val)
        f1_hist.append(f1)
        f1_val_hist.append(f1_val)
        
        # TODO: should I save based on loss of train or change it to val loss ????
        # Save the model checkpoint only if the current loss is better (lower) than the best loss
        if loss < best_loss: 
            best_loss = loss  # Update the best loss
            # save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_epoch_{epoch}.pth")
            save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_loss.pth")
            print(f"New best model saved with loss {loss:.4f} at epoch {epoch}")

        # Save model if loss has not imporved but accuracy has gotten better
        if acc_val > best_acc:
            best_acc = acc_val
            save_model_checkpoint(model, optimizer, epoch, loss, acc, save_path, file_name=f"best_model_checkpoint_acc.pth")
            print(f"New best model saved with Acc {acc_val:.4f} at epoch {epoch}")

        
        if en_shcheduler == True:
            scheduler.step()
        if (epoch+1) % step_size == 0 :
            log_handle.info(f"Lerning rate updated because of scheduler, lr={optimizer.param_groups[0]['lr']}")
            print(f"Lerning rate updated because of scheduler, lr={optimizer.param_groups[0]['lr']}")

        # save acc and loss plot each 50 epochs
        if epoch % 10 == 0  and epoch != 0:
            save_training_plots(loss_hist, acc_hist, save_path)
            save_training_plots(loss_val_hist, acc_val_hist, save_path,file_name_prefix="validation")
            save_training_plots(f1_hist, f1_val_hist, save_path,file_name_prefix="F1_Only",title1="F1_train", title2="F1_val")

    # Load best acc models
    model_load_path_acc = os.path.join(save_path,"best_model_checkpoint_acc.pth")
    checkpoint = torch.load(model_load_path_acc)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_test_acc, acc_test_acc, f1_test_acc = validation_with_tqdm_withF1(model,test_loader, loss_fn, device, is_binary, num_classes)

    # Load best loss models
    model_load_path_acc = os.path.join(save_path,"best_model_checkpoint_loss.pth")
    checkpoint = torch.load(model_load_path_acc)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_test_loss, acc_test_loss, f1_test_loss = validation_with_tqdm_withF1(model,test_loader, loss_fn, device, is_binary, num_classes)


    # Test model performance on test data
    # loss_test, acc_test, f1_test = validation_with_tqdm_withF1(model,test_loader, loss_fn, device, is_binary, num_classes)
    # log_handle.info(f"[Test] Loss: {loss_test} , Accuracy: {acc_test}, F1-Score: {f1_test}")

    log_handle.info(f"[Test-Best Loss] Loss: {loss_test_loss} , Accuracy: {acc_test_loss}, F1-Score: {f1_test_loss}")
    log_handle.info(f"[Test-Best Accuracy] Loss: {loss_test_acc} , Accuracy: {acc_test_acc}, F1-Score: {f1_test_acc}")

    # Save the training plots
    save_training_plots(loss_hist, acc_hist, save_path)
    save_training_plots(loss_val_hist, acc_val_hist, save_path,file_name_prefix="validation")
    save_training_plots(f1_hist, f1_val_hist, save_path,file_name_prefix="F1_Only",title1="F1_train", title2="F1_val")

    
    log_handle.info(f"[BEST ACC] Train: {max(acc_hist)} , Validation: {max(acc_val_hist)} , Test: {max(acc_test_loss,acc_test_acc)} ")
    log_handle.info(f"[BEST Loss] Train: {min(loss_hist)} , Validation: {min(loss_val_hist)} , Test: {min(loss_test_loss,loss_test_acc)} ")
    log_handle.info(f"[BEST F1] Train: {max(f1_hist)} , Validation: {max(f1_val_hist)} , Test: {max(f1_test_acc,f1_test_loss)} ")
    
    log_handle.info(f"Model Parameter Count: {get_num_params(model,1)} ")
    log_handle.info("DONE!")

    print(f"[BEST ACC] Train: {max(acc_hist)} , Validation: {max(acc_val_hist)} , Test: {max(acc_test_loss,acc_test_acc)} ")
    print(f"[BEST Loss] Train: {min(loss_hist)} , Validation: {min(loss_val_hist)} , Test: {min(loss_test_loss,loss_test_acc)} ")
    print(f"[BEST F1] Train: {max(f1_hist)} , Validation: {max(f1_val_hist)} , Test: {max(f1_test_acc,f1_test_loss)} ")
    print(f"Model Parameter Count: {get_num_params(model,1)} ")
    print("Training complete and data saved!")

    return loss_hist, acc_hist , loss_val_hist , acc_val_hist , min(loss_test_loss,loss_test_acc), max(acc_test_loss,acc_test_acc) , (f1_hist, f1_val_hist, max(f1_test_acc,f1_test_loss))


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
    # tvt_save_acc_loss_f1(model, dataset_name, model_name, emotion_dim, dataloader, dataloader, dataloader, optimizer, loss_fn, device, num_epochs=3, is_binary= False, num_classes=2 , pre_path='.')


