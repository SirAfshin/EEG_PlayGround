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
    '''
    Important note: embed_dim should be devidable by n_heads
        => embed_dim % num_heads == 0
    '''
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


# [2023]
# EEG-Based Emotion Recognition by Convolutional Neural Network with Multi-Scale Kernels
# Multi-Scale Convolution
# input
class MultiScaleKernelConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv5x5 = nn.Conv2d(in_channels ,out_channels ,kernel_size=5 ,stride=1, padding='same' )#,padding=1)
        self.conv7x7 = nn.Conv2d(in_channels ,out_channels ,kernel_size=7 ,stride=1, padding='same' )#,padding=2)
        self.conv1x1 = nn.Conv2d(2*out_channels ,out_channels ,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels) # 2 times because of concatanation
        self.relu = nn.ReLU()

    def forward(self,x):
        x1 = self.conv5x5(x)
        x2 = self.conv7x7(x)
        x = torch.cat((x1,x2),dim=1) # Concat along channel axis
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiScaleConv(nn.Module):
    def __init__(self,input_dim=[6,18,18], out_channels=[16,32,64,128],n_classes=2):
        super().__init__()
        self.multi_scale_convs = nn.ModuleList()
        
        in_channel = input_dim[0]
        for out_ch in out_channels:
            self.multi_scale_convs.append(
                MultiScaleKernelConvBlock(in_channel, out_ch))
            in_channel = out_ch
        
        self.fc = nn.Linear(out_channels[-1]*input_dim[1]*input_dim[2] , 2)

    def forward(self,x):
        for block in self.multi_scale_convs:
            x = block(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# TODO: make the tsception unet model
# TODO: make double conv of unet like the multi scale convs ? a Good change maybe
# UNET TSception ViT => to use with raw data
class DoubleConv_TSception(nn.Module):
    '''
    TSception inspired double conv block for unet basic blocks
    One layer of conv is Tceptions
    and the other layer is Sception
    '''
    def conv_block(self, in_chan, out_chan, kernel, step, padding=0):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan, out_channels=out_chan,
                kernel_size=kernel, stride=step, bias=False, padding=padding),
            # nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(inplace=True), # Or maybe relu
        )

    def __init__(self, in_channels, out_channels, num_T, sampling_rate=128, num_channels=14):
        super().__init__()
        # def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        self.num_S = out_channels
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[0] * sampling_rate) +1 ), 1, padding='same')
        self.Tception2 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[1] * sampling_rate) +1 ), 1, padding='same')
        self.Tception3 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[2] * sampling_rate) +1 ), 1, padding='same')

        self.fuse_T = nn.AvgPool2d((1,3),(1,3))

        self.Sception1 = self.conv_block(num_T, num_T, (int(num_channels), 1), 1 , )
        self.Sception2 = self.conv_block(num_T, num_T, (int(num_channels * 0.5), 1), (int(num_channels * 0.5), 1), )
       
        self.fusion_layer = self.conv_block(num_T, self.num_S, (1, 1), 1)
       
        self.adjust_height = nn.Conv2d(in_channels=self.num_S, out_channels=self.num_S,
                               kernel_size=(4,1), stride=(1,1), padding=(0,0))
       
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_T)
        self.BN_fusion = nn.BatchNorm2d(self.num_S)

        self.relu = nn.ReLU()

    def forward(self, x):
        # T-Kernels
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.relu(out)
        out = self.fuse_T(out)
        out1 = out # to use for residual connection

        # S-Kernels
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.relu(out)

        out = torch.cat((out,out1),dim=2) 

        # Fusion Layer
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = self.relu(out)
        
        # Adjust channel dim
        out = self.adjust_height(out)
        return out
         
class UNET_TSception(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64,128,256,512], 
                 num_T=16, sampling_rate= 128, num_channels= 14, pool_en= False
    ):
        super().__init__()
        self.pool_en= pool_en

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # TODO: do not apply pooling on channel dimension or maybe remove this pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Down part of Unet
        for feature in feature_channels:
            self.downs.append(DoubleConv_TSception(in_channels, feature, num_T, sampling_rate, num_channels))
            in_channels = feature
        
        # Up part of Unet
        for feature in reversed(feature_channels):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2), # *2 is for the concatanation
            ) # kernel=2 strid=2 -> doubles the height and width of image
            self.ups.append(DoubleConv_TSception(feature*2, feature , num_T, sampling_rate, num_channels))

        self.bottleneck = DoubleConv_TSception(feature_channels[-1], feature_channels[-1]*2 , num_T, sampling_rate, num_channels)
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections=[]

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            if self.pool_en == True:
                x = self.pool(x) # Remove if using for time data

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

# TODO: for now it only works with segment of 128 size
class UNET_TSception_classifier(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64,128,256,512], 
                 num_T=16, sampling_rate= 128, num_channels= 14, n_classes=2):
        super().__init__()
        self.ts_unet = UNET_TSception(in_channels, out_channels, feature_channels, num_T, sampling_rate, num_channels)
        self.fc = nn.Linear(out_channels * num_channels * sampling_rate ,n_classes)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.ts_unet(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = self.relu(x)
        return x

# TODO: Vit does not work with [channel, time] data I need to work sth out!
class UNET_VIT_TSception(nn.Module): 
    '''
    Important note: embed_dim should be devidable by n_heads
        => embed_dim % num_heads == 0
    '''
    def __init__(self,in_channels=128,unet_out_channels=3,img_size=9, patch_size=3, 
    n_classes=2,embed_dim=768,depth=5, n_heads=6,mlp_ratio=4.,qkv_bias=True,p=0.5,attn_p=0.5,
    sampling_rate= 128, num_channels= 14):
        super().__init__()
        self.unet = UNET_TSception(
            in_channels=in_channels, out_channels=unet_out_channels, feature_channels=[64,128,256,512], 
            num_T=16, sampling_rate= sampling_rate, num_channels= num_channels)

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

# ############################# UNET INCEPTION ################################

# [https://www.youtube.com/watch?v=uQc4Fs7yx5I]
class Inception_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))
        

#[https://www.youtube.com/watch?v=uQc4Fs7yx5I]
class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super().__init__()

        self.branch1 = Inception_Conv_Block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            Inception_Conv_Block(in_channels, red_3x3, kernel_size=1),
            Inception_Conv_Block(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            Inception_Conv_Block(in_channels, red_5x5, kernel_size=1),
            Inception_Conv_Block(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Inception_Conv_Block(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        # concat along filters dimension
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],dim=1) 

class DoubleConv_Inception(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=128 ,red_channels1=32, red_channels2=32):
        super().__init__()
        self.layer = nn.Sequential(
            Inception_Block(in_channels, out_1x1=hid_channels, red_3x3=red_channels1, out_3x3=hid_channels, red_5x5=red_channels1, out_5x5=hid_channels, out_1x1pool=hid_channels),
            nn.Conv2d(hid_channels*4, hid_channels, kernel_size=1),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(),
            Inception_Block(in_channels=hid_channels, out_1x1=out_channels, red_3x3=red_channels2, out_3x3=out_channels, red_5x5=red_channels2, out_5x5=out_channels, out_1x1pool=out_channels),
            nn.Conv2d(out_channels*4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class UNET_INCEPTION(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64,128,256,512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of Unet
        for feature in feature_channels:
            self.downs.append(DoubleConv_Inception(in_channels, feature))
            in_channels = feature
        
        # Up part of Unet
        for feature in reversed(feature_channels):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2), # *2 is for the concatanation
            ) # kernel=2 strid=2 -> doubles the height and width of image
            self.ups.append(DoubleConv_Inception(feature*2, feature))

        self.bottleneck = DoubleConv_Inception(feature_channels[-1], feature_channels[-1]*2)
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
            x = self.ups[idx](x) # Conv Transpose 
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = resize_tensor(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along channel dimension (b,channel,h,w)
            x = self.ups[idx+1](concat_skip) # DoubleConv_Inception

        x = self.final_conv(x)
        
        return x

# Adding conv 1x1 to concat of unet
class UNET_INCEPTION_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64,128,256,512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of Unet
        for feature in feature_channels:
            self.downs.append(DoubleConv_Inception(in_channels, feature))
            in_channels = feature
        
        # Up part of Unet
        for feature in reversed(feature_channels):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2), # *2 is for the concatanation
            ) # kernel=2 strid=2 -> doubles the height and width of image
            self.ups.append(nn.Sequential(
                nn.Conv2d(feature,feature,kernel_size=(1,1)),
                nn.ReLU(),
            ))
            self.ups.append(DoubleConv_Inception(feature*2, feature))

        self.bottleneck = DoubleConv_Inception(feature_channels[-1], feature_channels[-1]*2)
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections=[]

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # reverse list of skip connections for the decoder part (up)

        for idx in range(0, len(self.ups), 3): # each up and double conv is a single step
            x = self.ups[idx](x) # Conv Transpose 
            skip_connection = skip_connections[idx//3]
            skip_connection = self.ups[idx+1](skip_connection)

            if x.shape != skip_connection.shape:
                x = resize_tensor(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along channel dimension (b,channel,h,w)
            x = self.ups[idx+2](concat_skip) # DoubleConv_Inception

        x = self.final_conv(x)
        
        return x



class UNET_VIT_INCEPTION(nn.Module):
    '''
    Important note: embed_dim should be devidable by n_heads
    => embed_dim % num_heads == 0
    '''
    def __init__(self,in_channels=128,unet_out_channels=3,img_size=9, patch_size=3, 
    n_classes=2,embed_dim=768,depth=5, n_heads=6,mlp_ratio=4.,qkv_bias=True,p=0.5,attn_p=0.5,):
        super().__init__()
        self.unet = UNET_INCEPTION(in_channels=in_channels,out_channels=unet_out_channels, feature_channels=[64,128,256,512])
        self.vit = VisionTransformerEEG(
            img_size= img_size, # data[b,n_channel,x,x] # x= 17, 22, ...
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

    print('*'*20)
    x = torch.rand(10,14,17,17)

    model = UNET(in_channels=x.shape[1],out_channels=1, feature_channels=[64,128,256,512])
    print(get_num_trainable_params(model,1))
    print(f"[UNET] original input: {x.shape}, output: {model(x).shape}")

    model = UNET_VIT(
        in_channels=x.shape[1],unet_out_channels=3,img_size=17, patch_size=3, 
        n_classes=2,embed_dim=256,depth=5, n_heads=8,mlp_ratio=4.,qkv_bias=True,p=0.5,attn_p=0.5)
    print(get_num_trainable_params(model,1))
    print(f"[UNET_VIT] original input: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,6,18,18)
    model = MultiScaleKernelConvBlock(6,12)
    model2 = MultiScaleConv()
    print(f"[MultiScaleKernelConvBlock] original: {x.shape}, output: {model(x).shape}")
    print(f"[MultiScaleKernelConv] original: {x.shape}, output: {model2(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,14,17,17)
    model = MultiScaleKernelConvBlock(x[0].shape[0],12)
    model2 = MultiScaleConv(input_dim= x[0].shape, out_channels=[16,32,64,128], n_classes=2)
    print(f"[MultiScaleKernelConvBlock] original: {x.shape}, output: {model(x).shape}")
    print(f"[MultiScaleKernelConv] original: {x.shape}, output: {model2(x).shape}")
    #########################################################################

    # print('*'*20)
    # x = torch.rand(10,1,14,128)
    # model = DoubleConv_TSception(1,1,16,128,14)
    # # model = DoubleConv(1,1)
    # model2 = UNET_TSception(1,3)
    # print(f"Num trainable params: {get_num_trainable_params(model,1)}")
    # print(f"[DoubleConv_TSception] original: {x.shape}, output: {model(x).shape}")
    # print(f"[DoubleConv_TSception] original: {x.shape}, output: {model(model(x)).shape}")
    # print(f"[UNET_TSception] original: {x.shape}, output: {model2(x).shape}")
    # # print(f"[MultiScaleKernelConv] original: {x.shape}, output: {model2(x).shape}")
    #########################################################################

    # print('*'*20)
    # x = torch.rand(10,14,22,22)
    # model = UNET_VIT_TSception(
    #     in_channels=x.shape[1],unet_out_channels=3,img_size=22, patch_size=3, 
    #     n_classes=2,embed_dim=256,depth=5, n_heads=8,mlp_ratio=4.,qkv_bias=True,p=0.5,attn_p=0.5,
    #     sampling_rate= 16, num_channels= 22
    # ) # Change sampling rate so that the Tsception kernels can have good kernel size 
    # # samplig rate /2(4 and 8)
    # print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    # print(f"[UNET_VIT_TSception] original: {x.shape}, output: {model(x).shape}")
    #########################################################################

    # print('*'*20)
    # x = torch.rand(10,1,14,128)
    # model = UNET_TSception_classifier(1,3)
    # print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    # print(f"[UNET_TSception_classifier] original: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,14,22,22)
    model = DoubleConv(14,128)
    print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    print(f"[DoubleConv] original: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,14,22,22)
    model = DoubleConv_Inception(14,128)
    print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    print(f"[DoubleConv_Inception] original: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,14,22,22)
    model = UNET(in_channels=x.shape[1],out_channels=3, feature_channels=[64,128,256,512])
    print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    print(f"[UNET] original input: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,14,22,22)
    model = UNET_INCEPTION(in_channels=14, out_channels=3, feature_channels=[64,128,256,512])
    print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    print(f"[UNET_INCEPTION] original: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,14,22,22)
    model = UNET_VIT(
        in_channels=14, unet_out_channels=3, img_size=22, patch_size=3, n_classes=2, 
        embed_dim=768, depth=5, n_heads=6, mlp_ratio=4.0, qkv_bias=True, p=0.5, attn_p=0.5)
    print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    print(f"[UNET_VIT] original: {x.shape}, output: {model(x).shape}")
    #########################################################################

    print('*'*20)
    x = torch.rand(10,14,22,22)
    model = UNET_VIT_INCEPTION(
        in_channels=14, unet_out_channels=3, img_size=22, patch_size=3, n_classes=2, 
        embed_dim=768, depth=5, n_heads=6, mlp_ratio=4.0, qkv_bias=True, p=0.5, attn_p=0.5)
    print(f"Trainable param count : {get_num_trainable_params(model,1)}")
    print(f"[UNET_VIT_INCEPTION] original: {x.shape}, output: {model(x).shape}")
    #########################################################################