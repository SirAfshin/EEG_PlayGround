import sys
import os

# Dynamically add the root directory to sys.path
# Assumes that 'models' and 'utils' are in the same project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch 
import torch.nn as nn
import math
from utils.utils import print_var


# Important note : in order to get the same output size with padding the kernel size must be odd in that dimension
class Two_Layer_CNN(nn.Module):
    def __init__(self, eeg_channel=14, ) -> None:
        super().__init__()
        first_layer_out_channel = 16
        output_channels = 128
        spatial_kernel_length = eeg_channel - 1

        self.conv_spatial = nn.Conv2d(in_channels=1, out_channels=first_layer_out_channel, \
                                        kernel_size=(spatial_kernel_length, 1), stride=1,\
                                        padding=( int(math.ceil((spatial_kernel_length-1)/2)) , 0 ))

        self.conv_temporal = nn.Conv2d(in_channels=1, out_channels=first_layer_out_channel, \
                                        kernel_size=(1,11), stride=1, \
                                        padding=( 0 , int(math.ceil((11 - 1)/2))) )

        self.conv_spatial_temporal = nn.Conv2d(in_channels=1, out_channels=first_layer_out_channel, \
                                                kernel_size=(3,3), stride=1, \
                                                padding=( int(math.ceil((3-1)/2)) , int(math.ceil((3-1)/2 ))) )
        
        self.conv_last = nn.Conv2d(in_channels=first_layer_out_channel*3, out_channels=output_channels, \
                                    kernel_size=(3,3), stride=1, \
                                    padding=( int(math.ceil((3-1)/2)) , int(math.ceil((3-1)/2 ))) )
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activation_func = nn.ReLU()
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((9,9))
        
        self.fc = nn.Linear(9*9*output_channels, 1)

    def forward(self, x):
        # print_var('x', x.shape)

        x1 = self.max_pool(self.activation_func(self.conv_spatial(x)))
        x2 = self.max_pool(self.activation_func(self.conv_temporal(x)))
        x3 = self.max_pool(self.activation_func(self.conv_spatial_temporal(x)))
        
        if(len(x.shape) == 3): # data is not in batch (channel, h, w)
            x = torch.cat((x1, x2, x3), dim=0)
        else: # data is in batch format  (batch, channel, h, w)
            x = torch.cat((x1, x2, x3), dim=1)

        x = self.activation_func(x)
        x = self.conv_last(x)
        x = self.adaptive_avg_pool_2d(x)

        if(len(x.shape) == 3): # data is not in batch (channel, h, w)
            x = torch.flatten(x)
        else: # data is in batch format  (batch, channel, h, w)
            x = torch.flatten(x,1) # flatten all dimensions except batch
    
        x = self.fc(x)
        x = nn.functional.sigmoid(x)
        # print_var('x after fc ', x.shape)

        return x
        # print_var('x1', x1.shape)
        # print_var('x2', x2.shape)
        # print_var('x3', x3.shape)
        # print_var('concat',x.shape)


class Two_Layer_CNN_Pro(nn.Module):
    def __init__(self, eeg_channel=14, dropout_prob=0.5) -> None:
        super().__init__()
        first_layer_out_channel = 16
        output_channels = 128
        spatial_kernel_length = eeg_channel - 1

        # Convolutional layers
        self.conv_spatial = nn.Conv2d(in_channels=1, out_channels=first_layer_out_channel, 
                                      kernel_size=(spatial_kernel_length, 1), stride=1,
                                      padding=(int(math.ceil((spatial_kernel_length - 1) / 2)), 0))

        self.conv_temporal = nn.Conv2d(in_channels=1, out_channels=first_layer_out_channel, 
                                       kernel_size=(1, 11), stride=1, 
                                       padding=(0, int(math.ceil((11 - 1) / 2))))

        self.conv_spatial_temporal = nn.Conv2d(in_channels=1, out_channels=first_layer_out_channel, 
                                               kernel_size=(3, 3), stride=1, 
                                               padding=(int(math.ceil((3 - 1) / 2)), int(math.ceil((3 - 1) / 2))))

        self.conv_last = nn.Conv2d(in_channels=first_layer_out_channel * 3, out_channels=output_channels, 
                                   kernel_size=(3, 3), stride=1, 
                                   padding=(int(math.ceil((3 - 1) / 2)), int(math.ceil((3 - 1) / 2))))

        # Batch Normalization
        self.bn_spatial = nn.BatchNorm2d(first_layer_out_channel)
        self.bn_temporal = nn.BatchNorm2d(first_layer_out_channel)
        self.bn_spatial_temporal = nn.BatchNorm2d(first_layer_out_channel)
        self.bn_last = nn.BatchNorm2d(output_channels)

        # Max Pooling and Dropout
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(dropout_prob)

        # LeakyReLU activation
        self.activation_func = nn.LeakyReLU(negative_slope=0.1)
        
        # Adaptive average pooling to (9,9)
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((5, 5))
        
        # Fully connected layer for final output
        self.fc = nn.Linear(5 * 5 * output_channels, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        # Apply convolution layers with BatchNorm and activation function
        x1 = self.max_pool(self.dropout(self.activation_func(self.bn_spatial(self.conv_spatial(x)))))
        x2 = self.max_pool(self.dropout(self.activation_func(self.bn_temporal(self.conv_temporal(x)))))
        x3 = self.max_pool(self.dropout(self.activation_func(self.bn_spatial_temporal(self.conv_spatial_temporal(x)))))

        # Concatenate along the channel dimension
        if len(x.shape) == 3:  # If data is not in batch format (channel, h, w)
            x = torch.cat((x1, x2, x3), dim=0)
        else:  # Data is in batch format (batch, channel, h, w)
            x = torch.cat((x1, x2, x3), dim=1)

        # Apply final convolutional layer with BatchNorm
        x = self.activation_func(self.bn_last(self.conv_last(x)))

        # Apply adaptive average pooling
        x = self.adaptive_avg_pool_2d(x)

        # Flatten the tensor for the fully connected layer
        if len(x.shape) == 3:  # Data is not in batch format
            x = torch.flatten(x)
        else:  # Data is in batch format
            x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Fully connected output layer
        x = self.fc(x)
        x = self.fc2(x)

        return x

class Simplified_CNN(nn.Module):
    def __init__(self, eeg_channel=14, dropout_prob=0.5) -> None:
        super().__init__()
        first_layer_out_channel = 16
        output_channels = 128
        kernel_size = (eeg_channel, 3)  # Using (eeg_channel, 3) for simplicity in this single CNN layer

        # Single convolutional layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=first_layer_out_channel, 
                              kernel_size=kernel_size, stride=1, 
                              padding=(int(math.ceil((kernel_size[0] - 1) / 2)), int(math.ceil((kernel_size[1] - 1) / 2))))

        # Batch Normalization
        self.bn = nn.BatchNorm2d(first_layer_out_channel)

        # Max Pooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # LeakyReLU activation
        self.activation_func = nn.LeakyReLU(negative_slope=0.1)

        # Adaptive average pooling to (9,9)
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((9, 9))

        # Fully connected layer for final output
        self.fc = nn.Linear(9 * 9 * first_layer_out_channel, 1)

    def forward(self, x):
        # Apply the convolution, batch normalization, activation, and dropout
        x = self.dropout(self.activation_func(self.bn(self.conv(x))))

        # Apply max pooling
        x = self.max_pool(x)

        # Apply adaptive average pooling
        x = self.adaptive_avg_pool_2d(x)

        # Flatten the tensor for the fully connected layer
        if len(x.shape) == 3:  # If data is not in batch format (channel, h, w)
            x = torch.flatten(x)
        else:  # If data is in batch format (batch, channel, h, w)
            x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Fully connected output layer
        x = self.fc(x)

        return x


class EEGClassifierCNN(nn.Module):
    def __init__(self):
        super(EEGClassifierCNN, self).__init__()
        
        # Define a simple CNN with 2D convolutions
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Adjust the input size of the first fully connected layer
        self.fc1 = nn.Linear(32 * 2 * 2, 64)  # Now it's 32 * 2 * 2 = 128, matching the flattened size
        self.fc2 = nn.Linear(64, 2)  # Binary classification (2 classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        
        x = self.relu(self.conv1(x))  # Conv1 + ReLU
        x = self.pool(x)  # Pooling
        # print(f"Shape after conv1 and pooling: {x.shape}")
        
        x = self.relu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool(x)  # Pooling
        # print(f"Shape after conv2 and pooling: {x.shape}")
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        # print(f"Shape after flattening: {x.shape}")
        
        x = self.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)  # Fully connected layer 2
        
        return self.softmax(x)




if __name__ == "__main__":
    model = Two_Layer_CNN()
    
    data = torch.randn(10,1,14,128) # (batch, channel, height, width)
    target = torch.randn(1,1)
    # dataset = TensorDataset(data, target)
    # dataloader = DataLoader(dataset, batch_size= 10, shuffle= True)
    # print(len(dataloader))

    model(data)