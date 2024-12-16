import torch
import torch.nn as nn
from torcheeg.models import VanillaTransformer

class VanillaTransformer_time(nn.Module):
    def __init__(self,num_classes=1, num_electrodes=14):
        super().__init__()
        self.transformer = VanillaTransformer( num_electrodes= num_electrodes,
                                                chunk_size= 128,
                                                t_patch_size = 32,
                                                hid_channels = 32,
                                                depth = 3,
                                                heads = 4,
                                                head_channels = 8,
                                                mlp_channels= 64,
                                                num_classes = num_classes)

    def forward(self,x):
        x = self.transformer(x)
        x = nn.functional.sigmoid(x)
        return x
