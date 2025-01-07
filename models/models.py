import math

import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from einops.layers.torch import Rearrange

# Activation Function dictionary
nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())


# ************************************************ Modules *************************************************
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True):
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, bias=bias)
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class _TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float, activation: str = "relu"):
        super(_TCNBlock, self).__init__()
        #
        nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())
        #
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nonlinearity_dict[activation]
        self.drop1 = nn.Dropout(dropout)
        #
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nonlinearity_dict[activation]
        self.drop2 = nn.Dropout(dropout)
        #
        if in_channels != out_channels:
            self.project_channels = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.project_channels = nn.Identity()
        #
        self.final_nonlinearity = nonlinearity_dict[activation]

    def forward(self, x):
        # print(f"TCNBLOCK input: {x.shape}")
        residual = self.project_channels(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.drop2(out)
        # print(f"TCNBLOCK output: {self.final_nonlinearity(out + residual).shape}")
        return self.final_nonlinearity(out + residual)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # print(f"MultiHeadSelfAttention input: {x.shape}")
        batch_size, d, T_w = x.size()

        # Permute to (batch_size, T_w, d)
        x = x.permute(0, 2, 1)

        # Linear transformations
        Q = self.query_linear(x)  # (batch_size, T_w, d_model)
        K = self.key_linear(x)  # (batch_size, T_w, d_model)
        V = self.value_linear(x)  # (batch_size, T_w, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)
        K = K.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)
        V = V.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, num_heads, T_w, T_w)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, T_w, T_w)
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, T_w, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, T_w, self.d_model)  # (batch_size, T_w, d_model)

        # Final linear transformation
        output = self.out_linear(context)  # (batch_size, T_w, d_model)
        output = self.layer_norm(output + x)  # Add & Norm

        # Permute back to (batch_size, d, T_w)
        output = output.permute(0, 2, 1)

        # print(f"MultiHeadSelfAttention Output: {x.shape}")
        return output


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, depth_multiplier, kernel_size, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depth_multiplier = depth_multiplier
        # depth_multiplier: Determines how many output channels each input channel should be expanded to
        self.depthwise = nn.Conv2d(
            in_channels, in_channels * depth_multiplier, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding=0  # Set padding to 0 to reduce height
        ) # When groups is set to in_channels it is a depthwise conv

    def forward(self, x):
        return self.depthwise(x)

class ATC_Conv(nn.Module):
    def __init__(self, n_channel, in_channels, F1, D, KC, P2, dropout=0.3):
        super(ATC_Conv, self).__init__()
        F2 = F1 * D  # Output dimension

        # The first layer of regular convolution: temporal convolution layer
        self.temporal_conv = nn.Conv2d(in_channels, F1, (1, KC), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer: depth convolution layer
        self.depthwise_conv = DepthwiseConv2d(in_channels=F1, depth_multiplier=D, kernel_size=(n_channel, 1))
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.avgpool1 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # The third convolution layer
        self.spatial_conv = nn.Conv2d(F1 * D, F2, (1, KC), padding='same', bias=False)  # Modify the kernel size to fit KC
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, P2))  # The pooling size is controlled by P2
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        # Temporal Convolution
        x = self.temporal_conv(x)
        x = self.batchnorm1(x)
        x = self.elu(x)
        x = self.dropout1(x)
        # Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout2(x)
        # Spatial Convolution
        x = self.spatial_conv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout3(x)
        return x

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


# **************************************** Other Models and Blocks *************************************************
# resnet basic block
class BasicBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# resnet bottleneck block
class Bottleneck(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html 
    # It is good if <<planes * 4 == inplanes>>
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # To reduce channesl
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # Extract spatial information
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False) # To restore channels to original number or ...
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        print(f'res: {residual.shape}')
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        print(f'last out: {out.shape}')
        out += residual
        out = self.relu(out)
        return out



# ************************************************ My Model *************************************************
# NovModel is a placeholder name
class NovModel(nn.Module):
    """Some Information about MyModule"""
    def __init__(self,F1= 14, layers_tcn=4, filt_tcn= 14, kernel_tcn=4, dropout_tcn= 0.5, activation_tcn= 'relu',
                 temporal_size=128, num_electrodes=14, layers_cheby=2, hid_channels_cheby=64, num_classes=2):
        super(NovModel, self).__init__()
        
        # Activation Function dictionary
        nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())

        # TCN Layers
        in_channels = [F1] + (layers_tcn - 1) * [filt_tcn] 
        dilations = [2 ** i for i in range(layers_tcn)]
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt_tcn, kernel_size=kernel_tcn, dilation=dilation,
                        dropout=dropout_tcn, activation=activation_tcn)
            for in_ch, dilation in zip(in_channels, dilations)
        ])
        self.BN1 = nn.BatchNorm1d(filt_tcn)


        # Chebynet - DGCNN
        self.graph_layer = Chebynet(temporal_size, layers_cheby, hid_channels_cheby)

        # Adjacency Matrix
        self.A = nn.Parameter(torch.FloatTensor(filt_tcn, filt_tcn))
        nn.init.xavier_normal_(self.A)


        # Classifier head
        self.fc1 = Linear(filt_tcn * hid_channels_cheby, 64)
        self.fc2 = Linear(64, num_classes) 


    def forward(self, x):
        x = x.squeeze(1)
        for blk in self.tcn_blocks:
            x = blk(x)
        x = self.BN1(x)
        L = normalize_A(self.A)
        result = self.graph_layer(x, L)

        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)

        return result

    # def __init__(self,
    #              in_channels: int = 5,
    #              num_electrodes: int = 62,
    #              num_layers: int = 2,
    #              hid_channels: int = 32,
    #              num_classes: int = 2):
    #     super(DGCNN, self).__init__()
    #     self.in_channels = in_channels
    #     self.num_electrodes = num_electrodes
    #     self.hid_channels = hid_channels
    #     self.num_layers = num_layers
    #     self.num_classes = num_classes

    #     self.layer1 = Chebynet(in_channels, num_layers, hid_channels)
    #     self.BN1 = nn.BatchNorm1d(in_channels)
    #     self.fc1 = Linear(num_electrodes * hid_channels, 64)
    #     self.fc2 = Linear(64, num_classes)
    #     self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
    #     nn.init.xavier_normal_(self.A)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
    #     L = normalize_A(self.A)
    #     result = self.layer1(x, L)
    #     result = result.reshape(x.shape[0], -1)
    #     result = F.relu(self.fc1(result))
    #     result = self.fc2(result)
    #     return result



# ***********************************************************************************************************
if __name__ == "__main__":
    x = torch.rand(10,14,128)
    model = NovModel()
    print(model(x).shape)

##########################################################################
    # F2 = 14
    # layers = 4
    # filt = 14
    # kernel_s = 4
    # dropout = 0.5
    # activation = 'relu'

    # in_channels = [F2] + (layers - 1) * [filt] 
    # dilations = [2 ** i for i in range(layers)]
    # tcn_blocks = nn.ModuleList([
    #     _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
    #                 dropout=dropout, activation=activation)
    #     for in_ch, dilation in zip(in_channels, dilations)
    # ])

    # x = torch.rand(10,14,128)
    # print(x[0][0][0])
    # print(f"Start: {x.shape}")
    # for blk in tcn_blocks:
    #     x = blk(x)
    #     print(f"After blk: {x.shape}")
    #     print(x[0][0][0])

###################################################################
    # model = BasicBlock(1,10)
    # print(model(torch.rand(1,1,280,280)).shape)

    # model = Bottleneck(256,64)
    # print(model(torch.rand(1,256,280,280)).shape)

###################################################################
    # criterion = nn.CrossEntropyLoss() # For classification
    # criterion = nn.L1Loss() # For regression
    # criterion = nn.MSELoss() # For regression

    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)