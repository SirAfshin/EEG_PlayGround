import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# Activation Function dictionary
nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())

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
        residual = self.project_channels(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.drop2(out)
        return self.final_nonlinearity(out + residual)


# [2020] [EEG TCNet]
# [https://ieeexplore.ieee.org/document/9283028]
# Eeg-tcnet: An accurate temporal convo lutional network for embedded motor-imagery brain–machine interfaces
class EEGTCNet(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 32, layers: int = 2, kernel_s: int = 4, filt: int = 12,
                 dropout: float = 0.3, activation: str = 'relu', F1: int = 8, D: int = 2, kernLength: int = 32,
                 dropout_eeg: float = 0.2
                 ):
        super(EEGTCNet, self).__init__()
        regRate = 0.25
        numFilters = F1
        F2 = numFilters * D

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=0.001),
            Conv2dWithConstraint(F1, F2, (in_channels, 1), bias=False, groups=F1, max_norm=1),# depthwise separable convolution, where each input channel is processed independently.
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False), # depthwise separable convolution,
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            Rearrange("b c 1 t -> b c t")
        )

        in_channels = [F2] + (layers - 1) * [filt] # create list of input channels with first one being F2 and the rest are filt sized
        dilations = [2 ** i for i in range(layers)] # create dialation list with length of layers
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
                      dropout=dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])

        self.classifier = LinearWithConstraint(filt, n_classes, max_norm=regRate)

        # Initialization function for weights
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if hasattr(module, "weight") and module.weight is not None:
                if "norm" not in module.__class__.__name__.lower():
                    init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.eegnet(x)
        for blk in self.tcn_blocks:
            x = blk(x)

        x = self.classifier(x[:, :, -1])
        return x


class EEGNetModule(nn.Module):
    def __init__(self, channels, F1, D, kernLength, dropout, input_size):
        super(EEGNetModule, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = D * F1
        self.T = input_size[2]
        self.kernLength = int(kernLength)

        # Phase 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(channels, self.kernLength), groups=1,
                               padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=F1)
        # Depthwise Convolution
        self.depthwiseConv = nn.Conv2d(in_channels=F1, out_channels=self.F2, kernel_size=(channels, 1), groups=F1,
                                       bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=self.F2)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        # Separable Convolution
        self.separableConv = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=(1, 16), padding='same',
                                       bias=False, groups=self.F2)
        self.conv2 = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=self.F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.activation(self.batchnorm2(self.depthwiseConv(x)))
        x = self.dropout1(self.avg_pool1(x))
        x = self.separableConv(x)
        x = self.conv2(x)
        x = self.activation(self.batchnorm3(x))
        x = self.dropout2(self.avg_pool2(x))
        return x


# TCNet_Fusion
# [Electroencephalography-based motor imagery classification using temporal convolutional network fusion]
# (https://www.sciencedirect.com/science/article/abs/pii/S1746809421004237)
class TCNet_Fusion(nn.Module):
    def __init__(self, input_size, n_classes, channels, sampling_rate, kernel_s=3,
                 dropout=0.3, F1=24, D=2, dropout_eeg=0.3, layers=1, filt=12, activation='elu'):
        super(TCNet_Fusion, self).__init__()
        self.kernLength = int(0.25 * sampling_rate)
        F2 = F1 * D
        self.n_classes = n_classes

        self.EEGNet_sep = EEGNetModule(channels=channels, F1=F1, D=D, kernLength=self.kernLength, dropout=dropout_eeg,
                                       input_size=input_size)

        in_channels = [F2] + (layers - 1) * [filt]
        dilations = [2 ** i for i in range(layers)]
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
                      dropout=dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])
        size = self.get_size_temporal(input_size)
        self.dense = nn.Linear(size[-1], n_classes)
        self.softmax = nn.Softmax(dim=1)

    def get_size_temporal(self, input_size):
        data = torch.randn((1, input_size[0], input_size[1], input_size[2]))
        x = self.EEGNet_sep(data)
        eeg_output = torch.squeeze(x, 2)
        for blk in self.tcn_blocks:
            tcn_output = blk(eeg_output)
        con1_output = torch.cat((eeg_output, tcn_output), dim=1)  # Join along feature dimensions
        fc1_output = torch.flatten(eeg_output, start_dim=1)
        fc2_output = torch.flatten(con1_output, start_dim=1)
        # Another Concatenation
        con2_output = torch.cat((fc1_output, fc2_output), dim=1)
        size = con2_output.size()
        return size

    def forward(self, x):
        x = self.EEGNet_sep(x)
        eeg_output = torch.squeeze(x, 2)


        # TODO: This is not right not all blocks are being used -> Change later! 
        # for blk in self.tcn_blocks:
        #     tcn_output = blk(eeg_output)
        # DONE: Bellow 
        tcn_output = torch.squeeze(x, 2)
        for blk in self.tcn_blocks:
            tcn_output = blk(tcn_output)

        con1_output = torch.cat((eeg_output, tcn_output), dim=1)  # Join along feature dimensions
        fc1_output = torch.flatten(eeg_output, start_dim=1)
        fc2_output = torch.flatten(con1_output, start_dim=1)
        # Another Concatenation
        con2_output = torch.cat((fc1_output, fc2_output), dim=1)
        # Dense and Softmax
        dense_output = self.dense(con2_output)
        output = self.softmax(dense_output)
        return output


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


# ATCNet
# [Physics-informed attention temporal convolutional network for EEG-based motor imagery classification]
# (https://ieeexplore.ieee.org/abstract/document/9852687/)
class ATCNet(nn.Module):
    def __init__(self, input_size, n_channel, n_classes, n_windows=8,
                 eegn_F1=24, eegn_D=2, eegn_kernelSize=50, eegn_poolSize=2, eegn_dropout=0.3, num_heads=2,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3, fuse='average',
                 activation='elu'):
        super(ATCNet, self).__init__()
        self.n_windows = n_windows
        self.conv_block = ATC_Conv(n_channel, 1, eegn_F1, eegn_D, eegn_kernelSize, eegn_poolSize, eegn_dropout)
        self.fuse = fuse

        in_channels = [eegn_F1 * eegn_D] + (tcn_depth - 1) * [tcn_filters]
        dilations = [2 ** i for i in range(tcn_depth)]

        self.attention_block = MultiHeadSelfAttention(eegn_F1 * eegn_D, num_heads)

        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, tcn_filters, kernel_size=tcn_kernelSize, dilation=dilation,
                      dropout=tcn_dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])
        self.fuse_layer = nn.Linear(tcn_filters, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # (64,1,32,800)

        # print(f"X0: {x.shape}") #########################################

        x = self.conv_block(x)  # (64,100,1,12)

        # print(f"X1: {x.shape}") #########################################

        x = torch.flatten(x, start_dim=2)  # Flatten the channel and height dimensions (64,100,12)

        # print(f"X2: {x.shape}") #########################################

        outputs = []
        for i in range(self.n_windows):

            # print(f"X3: {x.shape}") #########################################

            windows_data = x[:, :, i:x.shape[2] - self.n_windows + i + 1]  # Sliding window
            
            # print(f"WINDOWS DATA: {windows_data.shape}") #########################
 
            # Attention block
            tcn_input = self.attention_block(windows_data)  # (batch_size, channels, T_w)

            # print(f"TCN INPUT: {tcn_input.shape}") ########################

            for blk in self.tcn_blocks:
                tcn_output = blk(tcn_input)
                tcn_input = tcn_output
            # (64,32,5)
            tcn_output = tcn_output[:, :, -1]  # Last timestep
            outputs.append(tcn_output)
        # (64,32)
        if self.fuse == 'average':
            output = torch.mean(torch.stack(outputs, dim=1), dim=1)
        elif self.fuse == 'concat':
            output = torch.cat(outputs, dim=1)
        else:
            raise ValueError("Invalid fuse method")

        output = self.fuse_layer(output)
        output = self.softmax(output)
        return output


if __name__ == "__main__":
    channels = 32 # 14
    eeg_x = torch.rand(1,1,channels,128)
    bm = nn.Sequential(nn.Conv2d(1, 8, (1, 32), padding="same", bias=False),
            nn.BatchNorm2d(8, momentum=0.01, eps=0.001),)
    model = Conv2dWithConstraint(8, 16, (channels, 1), bias=False, groups=8,max_norm=1)
    
    x = bm(eeg_x)
    print(eeg_x.shape)
    print(x.shape)
    print(model(x).shape)
    print('1*'*20)

    ###################################
    a = torch.rand(5,2,1,3)
    rr = Rearrange("b c 1 t -> b c t")
    print(a.shape)
    print(rr(a).shape)
    print('2*'*20)
    
    ###################################
    F2,filt,layers = 32 , 64, 3
    in_channels =  [F2] + (layers - 1) * [filt]
    print(in_channels)
    print('3*'*20)

    ##################################
    model = EEGTCNet(n_classes=2)
    x = torch.rand(1,1,32,128)
    print(model(x).shape)
    print('4*'*20)

    ##################################
    x = torch.rand(1,1,32,128)
    model = TCNet_Fusion(input_size= x[0].shape, # The size should not contain batch size !
                         n_classes= 2, 
                         channels= 32, 
                         sampling_rate= 128)
    print(model(x).shape)
    
    # EEG_NET = EEGNetModule(channels=32, F1=24, D=2, kernLength=int(0.25* 128), dropout=0.3, input_size=x[0].shape)
    # print(EEG_NET(x).shape)

    print('5*'*20)
    ##################################
    x = torch.rand(1,10,20)
    q = nn.Linear(10,2)
    
    print(x.shape)
    x = x.permute(0,2,1)
    print(x.shape)
    print(q(x).shape)

    print('6*'*20)
    ##################################
    x = torch.rand(1,1,32,128)
    model = ATCNet(x.shape, 32 , n_classes=2, n_windows=8,
                   eegn_F1=24, eegn_D=2, eegn_kernelSize=50, eegn_poolSize=1, eegn_dropout=0.3, num_heads=2,
                   tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3, fuse='average',activation='elu')
    print(model(x).shape)
    print(model(x))
    print(nn.CrossEntropyLoss()(model(x), torch.Tensor([[1]])))

    