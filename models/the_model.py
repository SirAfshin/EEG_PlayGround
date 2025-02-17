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
from models.cnn_based import UNET_INCEPTION, UNET_INCEPTION_2
from models.graph_models import DGCNN_ATTENTION, DGCNN_ATTENTION_Multi_head, DGCNN_ATTENTION_Transformer, DGCNN_ATTENTION_Transformer_Parallel

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        if in_channels != 4*intermediate_channels or stride != 1:
            self.identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels, intermediate_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(intermediate_channels * 4))
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

#################################################################################################

class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def normalize_A(A: torch.Tensor, symmetry: bool=False) -> torch.Tensor:
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A: torch.Tensor, num_layers: int) -> torch.Tensor:
    support = []
    for i in range(num_layers):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(A.device))
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class Chebynet(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super(Chebynet, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        for i in range(num_layers):
            self.gc1.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result = result+  self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self,
                 in_channels: int = 128, # dimension of node features
                 num_electrodes: int = 14, # number of nodes (EEG Channels)
                 num_layers: int = 2,
                 hid_channels: int = 32,
                 num_classes: int = 2):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet(in_channels, num_layers, hid_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = Linear(num_electrodes * hid_channels, 64)
        self.fc2 = Linear(64, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
        nn.init.xavier_normal_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use transpose so that the normalizations happens on nodes(channels)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)
        return result


class UNET_DGCNN_INCEPTION(nn.Module):
    def __init__(self,in_channels=14,unet_out_channels=3, unet_feature_channels=[64,128,256,512], n_classes=2):
        super().__init__()
        self.graph_feature_size = 5
        
        self.unet = UNET_INCEPTION( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=unet_out_channels, feature_channels=unet_feature_channels)
        
        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN(in_channels=(self.graph_feature_size ** 2), num_electrodes=(unet_out_channels + in_channels), num_layers=2, hid_channels=32, num_classes=n_classes)

        # print(f"NUM PARAM UNET INCEPTION: {get_num_trainable_params(self.unet,1)}")
        # print(f"NUM PARAM DGCNN: {get_num_trainable_params(self.dgcnn,1)}")
    
    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x


class UNET_DGCNN_INCEPTION2(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, n_classes=2):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.unet = UNET_INCEPTION( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=in_channels, feature_channels=unet_feature_channels)
        
        self.channel_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN(in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers, hid_channels=32, num_classes=n_classes)

        # print(f"NUM PARAM UNET INCEPTION: {get_num_trainable_params(self.unet,1)}")
        # print(f"NUM PARAM DGCNN: {get_num_trainable_params(self.dgcnn,1)}")
    
    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.channel_fusion(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x


class UNET_DGCNN_INCEPTION_GAT(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.unet = UNET_INCEPTION( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=in_channels, feature_channels=unet_feature_channels)
        
        self.channel_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

        # print(f"NUM PARAM UNET INCEPTION: {get_num_trainable_params(self.unet,1)}")
        # print(f"NUM PARAM DGCNN: {get_num_trainable_params(self.dgcnn,1)}")
    
    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.channel_fusion(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x


class UNET_NO_DGCNN_INCEPTION_GAT(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.unet = UNET_INCEPTION( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=in_channels, feature_channels=unet_feature_channels)
        
        self.channel_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        self.fc = nn.Sequential(
            nn.Linear(self.graph_feature_size **2 * in_channels, linear_hid),
            nn.ReLU(),
            nn.Linear(linear_hid, n_classes))
    
    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.channel_fusion(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


class NO_UNET_DGCNN_INCEPTION_GAT(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.channel_fusion = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

    def forward(self,x):
        x = self.channel_fusion(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x

# Using GATConv with MultiHeadAttention
class UNET_DGCNN_INCEPTION_GAT_Multi_Head(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, num_heads=4, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.unet = UNET_INCEPTION( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=in_channels, feature_channels=unet_feature_channels)
        
        self.channel_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION_Multi_head(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_heads=num_heads, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.channel_fusion(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x


# Using TransformerConv with dgcnn unet
class UNET_DGCNN_INCEPTION_GAT_Transformer(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, num_heads=4, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.unet = UNET_INCEPTION( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=in_channels, feature_channels=unet_feature_channels)
        
        self.channel_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION_Transformer(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_heads=num_heads, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.channel_fusion(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x

class NO_UNET_With_DGCNN_INCEPTION_GAT_Transformer(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, num_heads=4, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.channel_fusion = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION_Transformer(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_heads=num_heads, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

    def forward(self,x):
        x = self.channel_fusion(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x

# UNET_DGCNN_Parallel

'''The'''
class UNET_DGCNN_INCEPTION_GAT_Transformer_Parallel(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, num_heads=4, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.unet = UNET_INCEPTION( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=in_channels, feature_channels=unet_feature_channels)
        
        self.channel_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION_Transformer_Parallel(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_heads=num_heads, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.channel_fusion(x)
        x = nn.functional.leaky_relu(x) ## added ReLU
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x

'''The'''
class DGCNN_INCEPTION_GAT_Transformer_Parallel(nn.Module):
    '''
    conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, num_heads=4, n_classes=2, dropout=0.5, bias=True, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.channel_fusion = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION_Transformer_Parallel(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_heads=num_heads, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

    def forward(self,x):
        x = self.channel_fusion(x)
        x = nn.functional.leaky_relu(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = self.dgcnn(x)
        return x

# Add relu and leaky relu between 2 models
class UNET_DGCNN_INCEPTION_GAT_Transformer_2(nn.Module):
    '''
    unet       -> concat       -> conv1x1    -> adaptive avg pool  -> dgcnn
    [b,c,x,x]  -> [b,c*2,x,x]  -> [b,c,x,x]  -> [b,c,x',x']        -> [b,2]
    '''
    def __init__(self,in_channels=14, unet_feature_channels=[64,128,256,512], graph_feature_size=5, dgcnn_layers=2, dgcnn_hid_channels=32, num_heads=4, n_classes=2, dropout=0.5, bias=False, linear_hid=64):
        super().__init__()
        self.graph_feature_size = graph_feature_size
        
        self.unet = UNET_INCEPTION_2( # [b,c,x,x]: x=17,22,33
            in_channels=in_channels,out_channels=in_channels, feature_channels=unet_feature_channels)
        
        self.channel_fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=(1,1))

        self.avg_pool_global = nn.AdaptiveAvgPool2d(output_size=self.graph_feature_size)

        # input of dgcnn should be [batch, num_electrodes, in_channels]
        self.dgcnn = DGCNN_ATTENTION_Transformer(
            in_channels=(self.graph_feature_size ** 2), num_electrodes=in_channels, num_layers=dgcnn_layers,
            hid_channels=dgcnn_hid_channels, num_heads=num_heads, num_classes=n_classes, dropout=dropout, bias=bias, linear_hid=linear_hid)

    def forward(self,x):
        x_ = self.unet(x)
        x = torch.cat((x,x_), dim=1) # residual connection
        x = self.channel_fusion(x)
        x = nn.functional.relu(x)
        x = self.avg_pool_global(x)
        x = torch.flatten(x,2)
        x = nn.functional.leaky_relu(x)
        x = self.dgcnn(x)
        return x



if __name__ == "__main__":
    x = torch.rand(10,14,22,22)
    model = block(14,8) # in_channels == out_channels * 4
    print(f"[residual block] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,128)
    model = DGCNN(in_channels=128, num_electrodes=14, num_layers=2, hid_channels=32, num_classes=2)
    print(f"[DGCNN] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,3,33,33)
    model = nn.AdaptiveAvgPool2d(output_size=5)
    print(f"[AdaptiveAvgPool2d] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = UNET_DGCNN_INCEPTION(in_channels=14, unet_out_channels=3, unet_feature_channels=[64,128,256], n_classes=2)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[UNET_DGCNN_INCEPTION] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = UNET_DGCNN_INCEPTION2(in_channels=14, unet_feature_channels=[64,128,256], graph_feature_size=5, n_classes=2)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[UNET_DGCNN_INCEPTION 2] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = UNET_NO_DGCNN_INCEPTION_GAT(in_channels=14, unet_feature_channels=[64,128,256], graph_feature_size=5, n_classes=2,linear_hid=64)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[UNET_NO_DGCNN_INCEPTION_GAT] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = NO_UNET_DGCNN_INCEPTION_GAT(in_channels=14, unet_feature_channels=[64,128,256], graph_feature_size=5, n_classes=2,linear_hid=64)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[NO_UNET_DGCNN_INCEPTION_GAT] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = UNET_DGCNN_INCEPTION_GAT_Multi_Head(in_channels=14, unet_feature_channels=[64,128,256], graph_feature_size=5, num_heads=4, n_classes=2, linear_hid=64)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[UNET_DGCNN_INCEPTION_GAT_Multi_Head] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = UNET_DGCNN_INCEPTION_GAT_Transformer(in_channels=14, unet_feature_channels=[64,128,256], graph_feature_size=5, num_heads=4, n_classes=2, linear_hid=64)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[UNET_DGCNN_INCEPTION_GAT_Transformer] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = UNET_DGCNN_INCEPTION_GAT_Transformer_2(in_channels=14, unet_feature_channels=[64,128,256], graph_feature_size=5, num_heads=4, n_classes=2, linear_hid=64)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[UNET_DGCNN_INCEPTION_GAT_Transformer_2] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################

    x = torch.rand(10,14,33,33)
    model = UNET_DGCNN_INCEPTION_GAT_Transformer_Parallel(in_channels=14, unet_feature_channels=[64,128,256], graph_feature_size=5, num_heads=4, n_classes=2, linear_hid=64)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[UNET_DGCNN_INCEPTION_GAT_Transformer_Parallel] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################



