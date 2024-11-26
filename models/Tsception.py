
import torch.nn as nn
from torcheeg.models import TSCeption


class TSCEPTIONModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tsception = TSCeption(num_electrodes=14,
                        num_classes=1,
                        num_T=15,
                        num_S=15,
                        in_channels=1,
                        hid_channels=32,
                        sampling_rate=128,
                        dropout=0.5)
        
    def forward(self, x):# [Batch, 14, 128]
        x = x.unsqueeze(1) # [batch, 1, 14, 128]
        x = self.tsception(x)
        return x