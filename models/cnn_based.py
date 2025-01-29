import sys
import os

# Dynamically add the root directory to sys.path
# Assumes that 'models' and 'utils' are in the same project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from einops.layers.torch import Rearrange

from utils.utils import get_num_trainable_params, resize_tensor
from models.Transformer import VisionTransformerEEG


# Activation Function dictionary
nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())

# TSception
# [Tsception: a deep learning framework for emotion detection using EEG]
# (https://ieeexplore.ieee.org/abstract/document/9206750/)
class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool * 0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out

class MultiHeadSelfAttention_EEG(nn.Module):
    def __init__(self, input_dim, d_model, seq_length, num_heads):
        '''
        No mask is use in this Scaled Dot Product design !
        input_dim = time_length
        seq_length = num_channels
        Note: most of the time d_model = input_dim
        '''
        super(MultiHeadSelfAttention_EEG, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
   
    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            scaled += mask
        attention = F.softmax(scaled, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention
        
    def forward(self,x):
        x = x.squeeze(1)
        batch_size, sequence_length, input_dim = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # (b, head, c, features)
        q, k, v = qkv.chunk(3, dim=-1) # Extract Q K V
        values, attention = self.scaled_dot_product(q, k, v, None)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        out = self.layer_norm(out + x)  # Add & Norm
        
        # plt.imshow(attention[0][5].detach().numpy(), cmap= 'autumn')
        # plt.colorbar() 
        # plt.show()
        
        return out

class TSceptionATN(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSceptionATN, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8

        # Multi Head Self Attention
        self.multi_head_self_attention = MultiHeadSelfAttention_EEG(input_size[-1], input_size[-1], input_size[1],num_heads=8)

        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool * 0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = self.multi_head_self_attention(x)
        x = x.unsqueeze(1)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out

# U-Net
# [https://www.youtube.com/watch?v=IHq1t7NxS8k]
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # bias=False in order to use batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # bias=False in order to use batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),            
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64,128,256,512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of Unet
        for feature in feature_channels:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of Unet
        for feature in reversed(feature_channels):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2), # *2 is for the concatanation
            ) # kernel=2 strid=2 -> doubles the height and width of image
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(feature_channels[-1], feature_channels[-1]*2)
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections=[]

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # reverse list of skip connections for the decoder part (up)

        for idx in range(0, len(self.ups), 2): # each up and double conv is a single step
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = resize_tensor(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along channel dimension (b,channel,h,w)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        
        return x

class UNET_VIT(nn.Module):
    def __init__(self,in_channels=128,unet_out_channels=3,img_size=9, patch_size=3, 
    n_classes=2,embed_dim=768,depth=5, n_heads=6,mlp_ratio=4.,qkv_bias=True,p=0.5,attn_p=0.5,):
        super().__init__()
        self.unet = UNET(in_channels=in_channels,out_channels=unet_out_channels, feature_channels=[64,128,256])
        self.vit = VisionTransformerEEG(
            img_size= img_size, # data[3,9,9]
            patch_size=patch_size,
            in_chans=unet_out_channels,
            n_classes=n_classes,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            p=p,
            attn_p=attn_p,)
    
    def forward(self,x):
        x = self.unet(x)
        x = self.vit(x)
        return x




if __name__ == "__main__":
    x = torch.rand(10,1,14,128)
    mhsa = MultiHeadSelfAttention_EEG(128,128,14,8)
    print(mhsa(x).shape)
    #########################################################################

    model = TSceptionATN(2,[1,14,128],128,32,32,64,0.3)   
    print(model(x).shape)
    print(get_num_trainable_params(model,1))
    #########################################################################
    
    print('*'*20)
    x = torch.rand(10,128,9,9)
    model = UNET(in_channels=x.shape[1],out_channels=1, feature_channels=[64,128,256])
    print(get_num_trainable_params(model,1))
    print(f"[UNET] original input: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,128,9,9)
    model = UNET_VIT(
    in_channels=128,unet_out_channels=3,img_size=9, patch_size=3, 
    n_classes=2,embed_dim=768,depth=5, n_heads=6,mlp_ratio=4.,qkv_bias=True,p=0.5,attn_p=0.5
    )
    print(get_num_trainable_params(model,1))
    print(f"[UNET_VIT] original input: {x.shape}, output: {model(x).shape}")
    #########################################################################
    
    print('*'*20)
    x = torch.rand(10,14,17,17)

    model = UNET(in_channels=x.shape[1],out_channels=1, feature_channels=[64,128,256,512])
    print(get_num_trainable_params(model,1))
    print(f"[UNET] original input: {x.shape}, output: {model(x).shape}")

    model = UNET_VIT(
        in_channels=x.shape[1],unet_out_channels=3,img_size=17, patch_size=3, 
        n_classes=2,embed_dim=768,depth=5, n_heads=6,mlp_ratio=4.,qkv_bias=True,p=0.5,attn_p=0.5)
    print(get_num_trainable_params(model,1))
    print(f"[UNET_VIT] original input: {x.shape}, output: {model(x).shape}")
    #########################################################################

