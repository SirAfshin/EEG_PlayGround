import sys
import os

# Dynamically add the root directory to sys.path
# Assumes that 'models' and 'utils' are in the same project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


from utils.utils import print_var

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, Length, features = 14 channels)

        # Initialize hidden state and cell state with random values
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0)) # ([10, 14, 64])

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.sigmoid(out)
        return out


def accuracy_score(one_hot_outputs, labels):
    _, predicted = torch.max(one_hot_outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    score = correct / total
    return score

if __name__ == "__main__":
    # LSTM : LxF
    x = torch.randn(10,128,14) # (batch, channel, height, width)
    y = torch.randn(10,1)

    model = LSTM(14,64,1,1)

    print(model(x))
    print(x.shape)
    print(model(x).shape)




    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Set the hyperparameters
    # input_size = 1
    # hidden_size = 512
    # num_layers = 2
    # num_classes = 1
    # learning_rate = 0.001
    # batch_size = 200
    # num_epochs = 100

    # # Convert the data to PyTorch tensors
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.float32)


    # # Create a TensorDataset and DataLoader for the training data
    # train_data = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # # Create a TensorDataset and DataLoader for the test data
    # test_data = TensorDataset(X_test, y_test)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # # Initialize the model
    # model = LSTM(input_size, hidden_size, num_layers, num_classes)

    # # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # # Train the model
    # losses_list=[]
    # for epoch in range(num_epochs):
    #     for i, (inputs, labels) in enumerate(train_loader):
    #         # Forward pass

    #         inputs = inputs.unsqueeze(2)
    #         outputs = model(inputs)
    #         labels = torch.argmax(labels, dim=1)
    #         loss = criterion(outputs, labels)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # Print statistics
    #         if (i+1) % 10 == 0:
    #             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    #             losses_list.append(loss.item())

    # # Evaluate the model on the test set
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for inputs_t, labels_t in test_loader:
    #         inputs_t = inputs_t.unsqueeze(2)
    #         outputs_t = model(inputs_t)

    #         labels_t=torch.argmax(labels_t,dim=1)

    #         predicted = torch.argmax(outputs_t, dim=1)
    #         total += labels_t.size(0)
    #         correct += (predicted == labels_t).sum().item()

    # print(f'Test Accuracy: {100 * correct / total:.2f}%')
   

