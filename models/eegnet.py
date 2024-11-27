
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheeg.models import EEGNet


# source : https://github.com/braindecode/braindecode/tree/master/braindecode


class EEGNet_Normal_data(nn.Module):
    def __init__(self, num_electrodes=14,dropout=0.5, kernel_1=64, kernel_2=16, F1=8, F2=16, D=2, num_classes=1):
        super().__init__()
        
        self.eeg_net = EEGNet(chunk_size=128,
                            num_electrodes= num_electrodes,
                            dropout= dropout,
                            kernel_1= kernel_1,
                            kernel_2= kernel_2,
                            F1= F1,
                            F2= F2,
                            D= D,
                            num_classes= num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.eeg_net(x)
        return x



# Good Impelimentation for learning purposes
# source: https://github.com/amrzhd/EEGNet/blob/main/EEGNet.py

'''
The EEGNet model architecture used in this project is detailed below:

Block 1: Two sequential convolutional steps are performed. First, F1 2D convolutional filters of size (1, 32) are applied to capture frequency information at 2Hz and above. Then, a Depthwise Convolution of size (C, 1) is used to learn a spatial filter. Batch Normalization and ELU nonlinearity are applied, followed by Dropout for regularization. An average pooling layer is used for dimensionality reduction.
Block 2: Separable Convolution is used, followed by Pointwise Convolutions. Average pooling is used for dimension reduction.
Classification Block: Features are passed directly to a softmax classification with N units, where N is the number of classes in the data.
For further details, refer to the original EEGNet implementation.
'''



class EEGNetModel(nn.Module): # EEGNET-8,2
    def __init__(self, chans=22, classes=4, time_points=1001, temp_kernel=32,
                 f1=16, f2=32, d=2, pk1=8, pk2=16, dropout_rate=0.5, max_norm1=1, max_norm2=0.25):
        super(EEGNetModel, self).__init__()
        # Calculating FC input features
        linear_size = (time_points//(pk1*pk2))*f2

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False), # Depthwise Conv
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16),  groups=f2, bias=False, padding='same'), # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False), # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layer
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


