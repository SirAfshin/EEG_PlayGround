import sys
import os

# Dynamically add the root directory to sys.path
# Assumes that 'models' and 'utils' are in the same project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils.utils import get_num_trainable_params

# DGCNN with GATConv
class GraphConvAttention(nn.Module):
    def __init__(self, node_dim, in_channels, out_channels, dropout=0.0, bias=False):
        super(GraphConvAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.dropout = dropout
        
        # Initialise the weight matrix as a parameter
        self.W_alphas = nn.Parameter(torch.rand(in_channels, node_dim)) # (d x n)
        self.W = nn.Parameter(torch.rand(in_channels, out_channels)) # (d x d')
        nn.init.xavier_normal_(self.W_alphas)
        nn.init.xavier_normal_(self.W)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        D_neg, A_hat = self.update_adj_d(adj)

        alphas = torch.matmul(D_neg, torch.matmul(x, self.W_alphas))
        alphas = nn.functional.leaky_relu(alphas)
        alphas = nn.functional.softmax(alphas, dim=-2) # dim = -1 ???? is it OK
        alphas = nn.functional.dropout(alphas, self.dropout, training=self.training)

        out = torch.matmul(A_hat, x)  # Feature propagation
        out = torch.matmul(alphas, out)  # Apply attention coefficients
        out = torch.matmul(out, self.W)
        out = nn.functional.relu(out)
        if self.bias is not None:
            out = out + self.bias
            
        return out
       
    def update_adj_d(self, adj):
        # A_hat = A + I (adding self-loops)
        A_hat = adj + torch.eye(adj.size(-1), device=adj.device)

        # Compute the degree matrix D
        D = torch.sum(A_hat, dim=-1)  # Sum along last dimension to get degrees

        # Create D^{-1}
        D_inv = torch.pow(D + 1e-8, -1.0)  # Add epsilon to avoid division by zero
        D_inv = torch.diag_embed(D_inv)  # Convert to diagonal matrix

        return D_inv , A_hat       

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


class Chebynet_ATTENTION(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int, node_dim=14, dropout=0.5, bias=False):
        super(Chebynet_ATTENTION, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        for i in range(num_layers):
            self.gc1.append(GraphConvAttention(node_dim=node_dim, in_channels=in_channels, out_channels=out_channels, dropout=dropout, bias=bias))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result = result+  self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

class DGCNN_ATTENTION(nn.Module):
    def __init__(self,
                 in_channels= 128, # dimension of node features
                 num_electrodes= 14, # number of nodes (EEG Channels)
                 num_layers= 2,
                 hid_channels= 32,
                 num_classes= 2,
                 dropout=0.5,
                 bias=False):

        super(DGCNN_ATTENTION, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet_ATTENTION(in_channels= in_channels, num_layers=num_layers, out_channels=hid_channels, node_dim=num_electrodes, dropout=dropout, bias=bias)
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





if __name__ == "__main__":
    # model = GraphConvAttention(3,4,1)
    # x = torch.rand(10,2,3)
    # adj = torch.rand(10,2,2)
    # model(x,adj)

    # model = GAT_Layer(F_IN=3, F_OUT=4, heads=1, )
    # x = torch.rand(10,2,3)
    # adj = torch.rand(10,2,2)
    # model(x,adj)

    # a = torch.arange(10).repeat(2,1)
    # print(a)
    ################################################################################################

    model = GraphConvAttention(node_dim=14, in_channels=5, out_channels=10, dropout=0.5, bias=True)
    x = torch.rand(10,14,5)
    adj = torch.rand(10,14,14)
    print(f"[GraphConvAttention] original: {x.shape},  output: {model(x, adj).shape}")
    print('*'*100)
    ################################################################################################

    
    model = DGCNN_ATTENTION(in_channels=5,num_electrodes=14, num_layers=2, hid_channels=32, num_classes=2, dropout=0.5, bias=True)
    x = torch.rand(10,14,5)
    print(f"Num trainable params: {get_num_trainable_params(model,1)}")
    print(f"[DGCNN_ATTENTION] original: {x.shape},  output: {model(x).shape}")
    print('*'*100)
    ################################################################################################
    
    