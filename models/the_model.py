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
from models.cnn_based import UNET_INCEPTION


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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

#TODO: add Graph Attention (GAT) to the DGCNN graph
#TODO: Fix GraphAttention it is making problem
class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0):
        super(GraphAttention, self).__init__()
        self.heads = heads
        self.out_channels = out_channels // heads
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(in_channels, self.out_channels * heads))
        self.attn = nn.Parameter(torch.Tensor(1, heads, self.out_channels))
        
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.attn)

    def forward(self, x, adj):
        B, N, _ = x.shape  # Batch, Nodes, Features
        x = torch.matmul(x, self.weight).view(B, N, self.heads, self.out_channels)

        attn_scores = torch.matmul(x, self.attn.transpose(1, 2)).squeeze(-1)
        # adj = adj.unsqueeze(0).unsqueeze(0).expand(B, self.heads, -1, -1)
        attn_scores = attn_scores.masked_fill(adj == 0, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = F.dropout(attn_scores, self.dropout, training=self.training)

        out = torch.einsum("bhnk,bhn->bnk", x, attn_scores)
        return F.elu(out)

class ChebynetGAT(nn.Module):
    def __init__(self, in_channels, num_layers, out_channels, heads=1):
        super(ChebynetGAT, self).__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(GraphAttention(in_channels, out_channels, heads=heads))

    def forward(self, x, adj):
        for layer in self.attention_layers:
            x = layer(x, adj) + x  # Residual connection
        return x

class DGCNN_GAT(nn.Module):
    '''
    in_channels:    dimension of node features
    num_electrodes: number of nodes (EEG Channels)
    '''
    def __init__(self, in_channels=128, num_electrodes=14, num_layers=2, hid_channels=32, num_classes=2, heads=4):
        super(DGCNN_GAT, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = ChebynetGAT(in_channels, num_layers, hid_channels, heads)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(num_electrodes * hid_channels, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
        nn.init.xavier_normal_(self.A)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        adj = normalize_A(self.A)
        result = self.layer1(x, adj)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)
        return result


#TODO: Maybe I need to use this?????
class GraphAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, bias=True):
        super(GraphAttentionConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # Linear transformation for multi-head attention
        self.W = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.a = nn.Linear(2 * out_channels, heads, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * heads))
        else:
            self.bias = None

        # Xavier Initialization
        nn.init.xavier_normal_(self.W.weight)
        nn.init.xavier_normal_(self.a.weight)

    def forward(self, x, adj):
        """
        x: Node feature matrix (batch_size, num_nodes, in_channels)
        adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation
        h = self.W(x)  # Shape: (batch_size, num_nodes, out_channels * heads)

        # Reshape for multi-head attention
        h = h.view(batch_size, num_nodes, self.heads, self.out_channels)
        h_src = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)  # (B, N, N, heads, out_channels)
        h_dst = h.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)  # (B, N, N, heads, out_channels)

        # Compute attention scores
        attn_input = torch.cat([h_src, h_dst], dim=-1)  # (B, N, N, heads, 2*out_channels)
        attn_scores = self.a(attn_input).squeeze(-1)  # (B, N, N, heads)

        # Mask attention scores using adjacency matrix
        attn_scores = attn_scores.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))  # (B, N, N, heads)

        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=2)  # (B, N, N, heads)

        # Weighted sum of neighbors
        h_prime = torch.einsum("bnnh,bnhc->bnhc", attn_weights, h)  # (B, N, heads, out_channels)

        # Concatenate multi-head outputs
        h_prime = h_prime.reshape(batch_size, num_nodes, -1)  # (B, N, heads * out_channels)

        # Apply bias if present
        if self.bias is not None:
            h_prime += self.bias

        return F.relu(h_prime)

#TODO: check this as well it is full impelimentation
from torch_geometric.nn import GATConv, DynamicEdgeConv

class DGCNN_GAT(nn.Module):
    def __init__(self, in_channels, num_electrodes, num_layers=2, hid_channels=32, num_classes=2, heads=4):
        super(DGCNN_GAT, self).__init__()
        self.num_layers = num_layers
        self.num_electrodes = num_electrodes
        
        # Dynamic Edge Convolution for feature-based graph construction
        self.edge_conv = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * in_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, hid_channels)
        ), k=4)  # k-nearest neighbors
        
        # GAT Layers for Graph Attention
        self.gat_layers = nn.ModuleList([
            GATConv(hid_channels, hid_channels, heads=heads, concat=True, dropout=0.2)
            for _ in range(num_layers)
        ])
        
        # Final classification layer
        self.fc = nn.Linear(hid_channels * heads, num_classes)
        
    def forward(self, x):
        batch_size, num_electrodes, in_channels = x.shape  # (B, N, C)
        x = x.view(batch_size * num_electrodes, in_channels)
        
        # Compute dynamic edges and features
        edge_index, edge_attr = self.edge_conv(x)
        
        # Apply GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.elu(x)  # Activation
        
        # Global Pooling (mean over electrodes)
        x = x.view(batch_size, num_electrodes, -1).mean(dim=1)
        
        # Classification
        out = self.fc(x)
        return out


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

    x = torch.rand(10,14,128)
    model = DGCNN_GAT(in_channels=128, num_electrodes=14, num_layers=2, hid_channels=32, num_classes=2, heads=4)
    print(f"Number of trainable parameters: {get_num_trainable_params(model,1)}")
    print(f"[DGCNN_GAT] original {x.shape} , output {model(x).shape}")
    print(100*'*')
    ###################################################################