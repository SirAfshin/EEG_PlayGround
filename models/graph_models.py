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


# DGCNN with TransformerConv multihead with parallel graph convs blocks like GLFANet
class DGCNN_ATTENTION_Transformer_Parallel(nn.Module):
    def __init__(self,
                 in_channels= 128, # dimension of node features
                 num_electrodes= 14, # number of nodes (EEG Channels)
                 num_layers= 2,
                 hid_channels= 32,
                 num_heads=4,
                 num_classes= 2,
                 dropout=0.5,
                 bias=False,
                 linear_hid=64):

        super(DGCNN_ATTENTION_Transformer_Parallel, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet_ATTENTION_Transformer(in_channels= in_channels, num_layers=num_layers, out_channels=hid_channels, node_dim=num_electrodes, num_heads=num_heads, dropout=dropout, bias=bias)
        self.layer2 = Chebynet_ATTENTION_Transformer(in_channels= in_channels, num_layers=num_layers, out_channels=hid_channels, node_dim=num_electrodes, num_heads=num_heads, dropout=dropout, bias=bias)
        
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = Linear(num_electrodes * hid_channels, linear_hid)
        self.fc2 = Linear(linear_hid, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
        nn.init.xavier_normal_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use transpose so that the normalizations happens on nodes(channels)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result2 = self.layer2(x,L)
        result = result + result2 # residual connection
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)
        return result



# DGCNN with TransformerConv multihead is a no brainer as well
class TransformerGraphConv(nn.Module):
    def __init__(self, node_dim, in_channels, out_channels, num_heads=4, dropout=0.0, bias=False):
        super(TransformerGraphConv, self).__init__()
        assert out_channels % num_heads == 0

        self.in_channels = in_channels
        self.out_channels = out_channels // num_heads  # Output dim per head
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Query, Key, Value transformation matrices
        self.W_q = nn.Parameter(torch.Tensor(num_heads, in_channels, self.out_channels))
        self.W_k = nn.Parameter(torch.Tensor(num_heads, in_channels, self.out_channels))
        self.W_v = nn.Parameter(torch.Tensor(num_heads, in_channels, self.out_channels))
        
        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_v)

        self.matching_conv = nn.Conv1d(in_channels, out_channels, 1)

        # Normalization and skip connection
        self.norm = nn.LayerNorm(out_channels)
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)
    
    def update_adj_d(self, adj):
        """ Compute the degree matrix and normalized adjacency matrix """
        A_hat = adj + torch.eye(adj.size(-1), device=adj.device)
        D = torch.sum(adj, dim=-1)
        D_neg = torch.pow(D, -1.0)  # D^(-1.0)
        D_neg[D_neg == float('inf')] = 0  # Handle divide by zero
        D_neg = torch.diag_embed(D_neg)
        return D_neg, A_hat
    
    def forward(self, x, adj):
        D_neg, A_hat = self.update_adj_d(adj)
        multi_head_out = []
        
        for i in range(self.num_heads):
            Q = torch.matmul(x, self.W_q[i])  # Query
            K = torch.matmul(x, self.W_k[i])  # Key
            V = torch.matmul(x, self.W_v[i])  # Value
            
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.out_channels ** 0.5)
            attn_scores = F.softmax(attn_scores, dim=-1)
            attn_scores = F.dropout(attn_scores, self.dropout, training=self.training)
            
            out = torch.matmul(attn_scores, V)
            out = F.relu(out)
            multi_head_out.append(out)
        
        out = torch.cat(multi_head_out, dim=-1)  # Concatenate across heads
        
        # Apply skip connection and layer norm
        x = self.matching_conv(x.transpose(-2, -1)).transpose(-2, -1)
        out = self.norm(out + x)
        
        if self.bias is not None:
            out = out + self.bias
            
        return out


class Chebynet_ATTENTION_Transformer(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int, node_dim=14, num_heads=4 ,dropout=0.5, bias=False):
        super(Chebynet_ATTENTION_Transformer, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        for i in range(num_layers):
            self.gc1.append(TransformerGraphConv(node_dim=node_dim, in_channels=in_channels, out_channels=out_channels, num_heads=num_heads,dropout=dropout, bias=bias))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result = result+  self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

class DGCNN_ATTENTION_Transformer(nn.Module):
    def __init__(self,
                 in_channels= 128, # dimension of node features
                 num_electrodes= 14, # number of nodes (EEG Channels)
                 num_layers= 2,
                 hid_channels= 32,
                 num_heads=4,
                 num_classes= 2,
                 dropout=0.5,
                 bias=False,
                 linear_hid=64):

        super(DGCNN_ATTENTION_Transformer, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet_ATTENTION_Transformer(in_channels= in_channels, num_layers=num_layers, out_channels=hid_channels, node_dim=num_electrodes, num_heads=num_heads, dropout=dropout, bias=bias)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = Linear(num_electrodes * hid_channels, linear_hid)
        self.fc2 = Linear(linear_hid, num_classes)
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








# DGCNN with GATConv with multi head attetion
class GraphConvAttention_Multi_Head(nn.Module):
    def __init__(self, node_dim, in_channels, out_channels, num_heads=4, dropout=0.0, bias=False):
        super(GraphConvAttention_Multi_Head, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels // num_heads  # Output dim per head
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention parameters
        self.W_alphas = nn.Parameter(torch.rand(num_heads, in_channels, node_dim))  # (h, d, n)
        self.W = nn.Parameter(torch.rand(num_heads, in_channels, self.out_channels))  # (h, d, d')
        
        nn.init.xavier_normal_(self.W_alphas)
        nn.init.xavier_normal_(self.W)
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)
    
    def update_adj_d(self, adj):
        """ Compute the degree matrix and normalized adjacency matrix """
        A_hat = adj+ torch.eye(adj.size(-1), device= adj.device)
        # D = torch.sum(adj, dim=-1, keepdim=True)  # Degree matrix (sum of rows)
        D = torch.sum(adj, dim=-1)  # Degree matrix (sum of rows)
       
        D_neg = torch.pow(D, -1.0) # D^(-1.0)
        # D_neg = torch.pow(D, -0.5)  # D^(-0.5)

        D_neg[D_neg == float('inf')] = 0  # Handle divide by zero
        D_neg = torch.diag_embed(D_neg)
        
        # A_hat = D_neg * A_hat * D_neg  # Normalized adjacency matrix
        
        return D_neg, A_hat
    
    def forward(self, x, adj):
        D_neg, A_hat = self.update_adj_d(adj)
        
        multi_head_out = []
        for i in range(self.num_heads):
            alphas = torch.matmul(D_neg, torch.matmul(x, self.W_alphas[i]))  # (N, n)
            alphas = F.leaky_relu(alphas)
            alphas = F.softmax(alphas, dim=-2)
            alphas = F.dropout(alphas, self.dropout, training=self.training)
            
            out = torch.matmul(A_hat, x)  # Feature propagation
            out = torch.matmul(alphas, out)  # Apply attention coefficients
            out = torch.matmul(out, self.W[i])  # Transform features
            out = F.relu(out)
            multi_head_out.append(out)
        
        out = torch.cat(multi_head_out, dim=-1)  # Concatenate across heads
        
        if self.bias is not None:
            out = out + self.bias
            
        return out

class Chebynet_ATTENTION_Multi_Head(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int, node_dim=14, num_heads=4 ,dropout=0.5, bias=False):
        super(Chebynet_ATTENTION_Multi_Head, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        for i in range(num_layers):
            self.gc1.append(GraphConvAttention_Multi_Head(node_dim=node_dim, in_channels=in_channels, out_channels=out_channels, num_heads=num_heads,dropout=dropout, bias=bias))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result = result+  self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

class DGCNN_ATTENTION_Multi_head(nn.Module):
    def __init__(self,
                 in_channels= 128, # dimension of node features
                 num_electrodes= 14, # number of nodes (EEG Channels)
                 num_layers= 2,
                 hid_channels= 32,
                 num_heads=4,
                 num_classes= 2,
                 dropout=0.5,
                 bias=False,
                 linear_hid=64):

        super(DGCNN_ATTENTION_Multi_head, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet_ATTENTION_Multi_Head(in_channels= in_channels, num_layers=num_layers, out_channels=hid_channels, node_dim=num_electrodes, num_heads=num_heads, dropout=dropout, bias=bias)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = Linear(num_electrodes * hid_channels, linear_hid)
        self.fc2 = Linear(linear_hid, num_classes)
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
                 bias=False,
                 linear_hid=64):

        super(DGCNN_ATTENTION, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet_ATTENTION(in_channels= in_channels, num_layers=num_layers, out_channels=hid_channels, node_dim=num_electrodes, dropout=dropout, bias=bias)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = Linear(num_electrodes * hid_channels, linear_hid)
        self.fc2 = Linear(linear_hid, num_classes)
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

##############################################################################################
################################ TO USE LATER IF IT IS HELPFULL ###############################
#TODO: add Graph Attention (GAT) to the DGCNN graph
#TODO: Fix GraphAttention it is making problem
class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, bias=False):
        super(GraphAttention, self).__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.out_channels = out_channels // heads
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(in_channels, self.out_channels * heads))
        self.attn = nn.Parameter(torch.Tensor(1, heads, self.out_channels))

        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.attn)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_channels * heads))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        B, N, _ = x.shape  # Batch, Nodes, Features
        x = torch.matmul(x, self.weight).view(B, N, self.heads, self.out_channels)

        attn_scores = torch.matmul(x, self.attn.transpose(1, 2)).squeeze(-1)
        # attn_scores = attn_scores.masked_fill(adj == 0, float('-inf'))
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


# GCN and GAT
# Source: [https://medium.com/@jrosseruk/demystifying-gcns-a-step-by-step-guide-to-building-a-graph-convolutional-network-layer-in-pytorch-09bf2e788a51]
class GCNLayer(nn.Module):
    """
        GCN layer

        Args:
            input_dim (int): Dimension of the input
            output_dim (int): Dimension of the output (a softmax distribution)
            A (torch.Tensor): 2D adjacency matrix
    """

    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # A_hat = A + I
        self.A_hat = self.A + torch.eye(self.A.size(0))

        # Create diagonal degree matrix D
        self.ones = torch.ones(input_dim, input_dim)
        self.D = torch.matmul(self.A.float(), self.ones.float())

        # Extract the diagonal elements
        self.D = torch.diag(self.D)

        # Create a new tensor with the diagonal elements and zeros elsewhere
        self.D = torch.diag_embed(self.D)
        
        # Create D^{-1/2}
        self.D_neg_sqrt = torch.diag_embed(torch.diag(torch.pow(self.D, -0.5)))
        
        # Initialise the weight matrix as a parameter
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, X: torch.Tensor):

        # D^-1/2 * (A_hat * D^-1/2)
        support_1 = torch.matmul(self.D_neg_sqrt, torch.matmul(self.A_hat, self.D_neg_sqrt))
        
        # (D^-1/2 * A_hat * D^-1/2) * (X * W)
        support_2 = torch.matmul(support_1, torch.matmul(X, self.W))
        
        # ReLU(D^-1/2 * A_hat * D^-1/2 * X * W)
        H = F.relu(support_2)

        return H


# Source: [https://github.com/arturtoshev/from_gcn_to_gat/blob/main/main.ipynb]

class GAT_Layer(nn.Module):   
    def __init__(self, F_IN: int, F_OUT: int, heads: int, concat: bool = True, 
                 negative_slope: float = 0.2, dropout_rate: float = 0.6):
        super().__init__()
        # TODO:     add_self_loops: bool = True  ???
        """ one GAT layer as in [gat, Sec. 3.3]

        Parameters
        ----------
        F_IN: int
            number of input features; equivalent to "F" in [gat, p.3]
            this is the result of `F_OUT*heads` from the previous layer
        F_OUT: int
            number of output features; equivalent to "F'" in [gat, p.3]
        heads: int
            number of attention heads
        concat: bool
            whether to concatenate the attention heads (True) or to average over them (False)
        negative_slope: float
            angle of negative slope in the LeakyReLU activation
        dropout_rate: float
            dropout rate
        """

        self.F_IN = F_IN
        self.F_OUT = F_OUT  
        self.K = heads
        self.concat = concat
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout_rate)

        self.lin = nn.Linear(self.F_IN, self.K * self.F_OUT)
        self.a_src = nn.Parameter(torch.Tensor(1, self.K, self.F_OUT))
        self.a_trg = nn.Parameter(torch.Tensor(1, self.K, self.F_OUT))

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.uniform_(self.lin.bias)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_trg)

    def forward(self, x, edge_index):
        """forward GAT function as in [gat, Sec. 3.3]

        Parameters
        ----------
        x : torch.tensor, shape(N, F_IN)
            feature matrix
        edge_index : torch.tensor, shape=(2, E)
            pairs of connected nodes excluding self-connections

        Returns
        -------
        torch.tensor, shape(N, F_OUT)
 
        Notes
        -----
        Scaling of the algorithm wrt number of nodes (N) and edges (E):
        1. compute W*h                  O(|N|)
        2. compute all attentions       O(|E|)
        3. normalize attentions         O(|N|*average_degree)
        4. aggregate weighted messages  O(|N|*average_degree)
        """

        N = x.shape[0]  # number of nodes
        # the input data excludes self-connections but [gat] uses them. So we need to add them
        edge_index_ = self.add_self_connections_to_edge_index(edge_index, N) # for my case A_hat = A + I 

        x = self.dropout(x)
        h_prime = self.lin(x).view(N, self.K, self.F_OUT)  # (N, F_IN)*(F_IN, K*F_OUT) -> (N, K*F_OUT) -> (N, K, F_OUT)

        # edge_index_[0] = source; edge_index_[1] = target
        h_src = (h_prime * self.a_src).sum(dim=2)  # ((N, K, F_OUT) * (1, K, F_OUT)).sum(2) -> (N, K)
        h_trg = (h_prime * self.a_trg).sum(dim=2)
        h_prime_edge_src = h_src[edge_index_[0]]  # (E, K)
        h_prime_edge_trg = h_trg[edge_index_[1]] 
        # compute raw attention coefficients `e_{ij}` in paper [gat, p.3]
        e = h_prime_edge_src + h_prime_edge_trg  # (E, K)
        e = self.leaky_relu(e)

        # apply softmax normalization over all source nodes per target node
        e = e - e.max()   # trick to improve numerical stability before computing exponents (for softmax)
        exp_e = e.exp()  # = unnormalized attention for each pair self.edge_pair[0]->self.edge_pair[1]
        trg_normalization = torch.zeros(size=(N, self.K), dtype=x.dtype, device=x.device)  # tensor with normalizing constants for each target node
        index = self.explicit_broadcast(edge_index_[1], exp_e)
        trg_normalization.scatter_add_(0, index, exp_e)   # index1:[E, K], exp_e:[E, K] -> [N, K]
        # In the above line we aggregate the coefficients `e_{ij}` to each target nodes j (in [gat] notation should be i, 
        # but here the second dimension is the target). A small demonstaration of what we do here would be: 
        # for K=2 heads and a graph of 3 nodes with connections 0-1-2:
        # edge_index = torch.tensor([[0,0,1,1,1,2,2],[0,1,0,1,2,1,1]])  # (2,7) = (2, E)
        # trg_index = edge_index[1].repeat(2,1).T                         # (7, 2) = (E, K)
        # out = torch.tensor([[0,1,2,3,4,5,6], [7,7,7,7,7,7,7]]).T    # (7, 2) = (E, K)
        # print(torch.zeros(3, 2, dtype=out.dtype).scatter_add_(0, trg_index, src)) # (3, 2) = (N, K)
        # >> tensor([[ 2, 14], [15, 28], [ 4,  7]])

        # normalized attention coefficients `alpha_{ij}` in paper [gat, p.3]
        alpha = exp_e / (trg_normalization[edge_index_[1]] + 1e-10)  # (E, K), s.t. for a given target, the sum of the sources = 1.
        # validate correctness for each target #node and #head by uncommenting the following line and plugging #node and #head
        # assert alpha[edge_index_[1]==#node, #head]) == 1.

        alpha = self.dropout(alpha)
        src = h_prime[edge_index_[0]]  # (E, K, F_OUT)
        src *= alpha.unsqueeze(-1) 
        out = torch.zeros(size=(N, self.K, self.F_OUT), dtype=x.dtype, device=x.device)
        index = self.explicit_broadcast(edge_index_[1], src)
        out = out.scatter_add(0, index, src)  # (N, K, F_OUT)  # h double prime

        if self.concat:
            out = out.view(N, -1)  # (N, K, F_OUT) -> (N, K*F_OUT)
        else:
            out = out.mean(dim=1)  # (N, K, F_OUT) -> (N, F_OUT)

        return out

    def add_self_connections_to_edge_index(self, edge_index, N):
        self_loop = torch.arange(N, dtype=int).repeat(2,1).to(device)
        edge_index_ = torch.cat((edge_index, self_loop), dim=1)
        return edge_index_

    def explicit_broadcast(self, this, other):
        """from https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(Cora).ipynb"""
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = GAT_Layer(num_features, 8, heads=8, dropout_rate=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GAT_Layer(8 * 8, num_classes, heads=1, concat=False, dropout_rate=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)



###############################################################################################


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
    
    model = GraphConvAttention_Multi_Head(node_dim=14,in_channels=5,out_channels=44,num_heads=4,dropout=0.5,bias=True)
    x = torch.rand(10,14,5)
    adj = torch.rand(10,14,14)
    print(f"Num trainable params: {get_num_trainable_params(model,1)}")
    print(f"[GraphConvAttention_Multi_Head] original: {x.shape},  output: {model(x,adj).shape}")
    print('*'*100)
    ################################################################################################


    model = DGCNN_ATTENTION_Multi_head(in_channels=5,num_electrodes=14, num_layers=2, hid_channels=32, num_classes=2, num_heads=4 ,dropout=0.5, bias=True)
    x = torch.rand(10,14,5)
    print(f"Num trainable params: {get_num_trainable_params(model,1)}")
    print(f"[DGCNN_ATTENTION_Multi_head] original: {x.shape},  output: {model(x).shape}")
    print('*'*100)
    ################################################################################################


    model = TransformerGraphConv(node_dim=14, in_channels=5, out_channels=12, num_heads=4, dropout=0.5, bias=True)
    x = torch.rand(10,14,5)
    adj = torch.rand(10,14,14)
    print(f"Num trainable params: {get_num_trainable_params(model,1)}")
    print(f"[TransformerGraphConv] original: {x.shape},  output: {model(x,adj).shape}")
    print('*'*100)
    ################################################################################################

    model = DGCNN_ATTENTION_Transformer(in_channels=5,num_electrodes=14, num_layers=2, hid_channels=32, num_classes=2, num_heads=4 ,dropout=0.5, bias=True)
    x = torch.rand(10,14,5)
    print(f"Num trainable params: {get_num_trainable_params(model,1)}")
    print(f"[DGCNN_ATTENTION_Transformer] original: {x.shape},  output: {model(x).shape}")
    print('*'*100)
    ################################################################################################

    model = DGCNN_ATTENTION_Transformer_Parallel(in_channels=5,num_electrodes=14, num_layers=2, hid_channels=32, num_classes=2, num_heads=4 ,dropout=0.5, bias=True)
    x = torch.rand(10,14,5)
    print(f"Num trainable params: {get_num_trainable_params(model,1)}")
    print(f"[DGCNN_ATTENTION_Transformer_Parallel] original: {x.shape},  output: {model(x).shape}")
    print('*'*100)
    ################################################################################################





