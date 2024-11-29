import sys
import os

# Dynamically add the root directory to sys.path
# Assumes that 'models' and 'utils' are in the same project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch 
import torch.nn as nn
import math
from utils.utils import print_var


class STFT_Two_Layer_CNN_Pro(nn.Module):
    def __init__(self, eeg_channel=14, dropout_prob=0.5) -> None:
        super().__init__()
        first_layer_out_channel = 16
        output_channels = 128
        spatial_kernel_length = eeg_channel - 1

        # Convolutional layers
        self.conv_spatial = nn.Conv2d(in_channels=14, out_channels=first_layer_out_channel, 
                                      kernel_size=(spatial_kernel_length, 1), stride=1,
                                      padding=(int(math.ceil((spatial_kernel_length - 1) / 2)), 0))

        self.conv_temporal = nn.Conv2d(in_channels=14, out_channels=first_layer_out_channel, 
                                       kernel_size=(1, 11), stride=1, 
                                       padding=(0, int(math.ceil((11 - 1) / 2))))

        self.conv_spatial_temporal = nn.Conv2d(in_channels=14, out_channels=first_layer_out_channel, 
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


if __name__ == "__main__":
    model = STFT_Two_Layer_CNN_Pro()
    
    data = torch.randn(10,14,33,5) # (batch, channel, height, width)
    target = torch.randn(1,1)
    # dataset = TensorDataset(data, target)
    # dataloader = DataLoader(dataset, batch_size= 10, shuffle= True)
    # print(len(dataloader))

    print(model(data).shape)
    