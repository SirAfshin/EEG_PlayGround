
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheeg.models import EEGNet


# source : https://github.com/braindecode/braindecode/tree/master/braindecode


class EEGNet_Normal_data(nn.Module):
    def __init__(self, num_electrodes=14,dropout=0.5, kernel_1=64, kernel_2=16, F1=8, F2=16, D=2, num_classes=1):
        super().__init__()
        
        self.eeg_net = EEGNet(chunk_size=128,
                            num_electrodes= num_electrodes,
                            dropout= dropout,
                            kernel_1= kernel_1,
                            kernel_2= kernel_2,
                            F1= F1,
                            F2= F2,
                            D= D,
                            num_classes= num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.eeg_net(x)
        return x





# source:  https://github.com/weilheim/EEG/blob/master/model/eegnet.py
class _EEGNet(nn.Module): ##  not working now
    """2D convolutional neural network for single EEG frame."""
    VALIDENDPOINT = ('logit', 'predict')

    def __init__(self, num_class,
                 input_channel,
                 hidden_size,
                 kernel_size,
                 stride,
                 avgpool_size=4,
                 dropout=0.1):
        super(EEGNet, self).__init__()

        assert len(kernel_size) == len(hidden_size)
        assert len(kernel_size) == len(stride)
        self.num_layer = len(kernel_size)
        self.num_class = num_class
        self.input_channel = input_channel
        self.dropout = dropout

        in_channel = self.input_channel
        layer = 1
        self.projections = nn.ModuleList()
        self.residualnorms = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for (out_channel, kernel_width, s) in zip(hidden_size, kernel_size, stride):
            pad = (kernel_size - 1) // 2
            self.projections.append(nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=s, bias=False)
                                    if in_channel != out_channel or s != 1 else None)
            self.residualnorms.append(nn.BatchNorm2d(out_channel)
                                      if in_channel != out_channel or s != 1 else None)
            self.convolutions.append(nn.Conv2d(in_channel, out_channel,
                                               kernel_size=kernel_width, stride=s, padding=pad, bias=False))
            self.batchnorms.append(nn.BatchNorm2d(out_channel, eps=1e-5, affine=True))
            in_channel = out_channel
        # avgpool_size should equal to size of the feature map,
        # otherwise self.predict will break.
        self.avgpool = nn.AvgPool2d(kernel_size=avgpool_size)
        self.predict = nn.Linear(hidden_size[-1], self.num_class, bias=True)

    def forward(self, x, endpoint='predict'):
        if endpoint not in self.VALIDENDPOINT:
            raise ValueError('Unknown endpoint {:s}'.format(endpoint))

        for proj, rbn, conv, bn in zip(self.projections, self.residualnorms,
                                  self.convolutions, self.batchnorms):
            if proj is not None:
                residual = proj(x)
                residual = rbn(residual)
            else:
                residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = bn(x)
            x = (x + residual)
            x = F.relu(x)
        x = self.avgpool(x)
        if endpoint == 'logit':
            return x
        x = self.predict(x)
        return x