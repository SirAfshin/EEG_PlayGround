'''
Code from torcheeg git repo
[https://github.com/torcheeg/torcheeg/blob/main/torcheeg/models/gnn/dgcnn.py]
'''
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


# DGCNN
# [EEG emotion recognition using dynamical graph convolutional neural networks]
# (https://ieeexplore.ieee.org/abstract/document/8320798)
class DGCNN(nn.Module):
    r'''
    Dynamical Graph Convolutional Neural Networks (DGCNN). For more details, please refer to the following information.

    - Paper: Song T, Zheng W, Song P, et al. EEG emotion recognition using dynamical graph convolutional neural networks[J]. IEEE Transactions on Affective Computing, 2018, 11(3): 532-541.
    - URL: https://ieeexplore.ieee.org/abstract/document/8320798
    - Related Project: https://github.com/xueyunlong12589/DGCNN

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.models import DGCNN
        from torcheeg.datasets import SEEDDataset
        from torcheeg import transforms

        dataset = SEEDDataset(root_path='./Preprocessed_EEG',
                              offline_transform=transforms.BandDifferentialEntropy(band_dict={
                                  "delta": [1, 4],
                                  "theta": [4, 8],
                                  "alpha": [8, 14],
                                  "beta": [14, 31],
                                  "gamma": [31, 49]
                              }),
                              online_transform=transforms.Compose([
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]))

        model = DGCNN(in_channels=5, num_electrodes=62, hid_channels=32, num_layers=2, num_classes=2)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`5`)
        num_electrodes (int): The number of electrodes. (default: :obj:`62`)
        num_layers (int): The number of graph convolutional layers. (default: :obj:`2`)
        hid_channels (int): The number of hidden nodes in the first fully connected layer. (default: :obj:`32`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self,
                 in_channels: int = 5,
                 num_electrodes: int = 62,
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
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 62, 5]`. Here, :obj:`n` corresponds to the batch size, :obj:`62` corresponds to :obj:`num_electrodes`, and :obj:`5` corresponds to :obj:`in_channels`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)
        return result


# LGGNet
# [LGGNet: Learning from local-global-graph representations for brain–computer interface]
# (https://ieeexplore.ieee.org/abstract/document/10025569)
class LGGNet(nn.Module):
    r'''
    DLocal-Global-Graph Networks (LGGNet). For more details, please refer to the following information.

    - Paper: Ding Y, Robinson N, Zeng Q, et al. LGGNet: learning from Local-global-graph representations for brain-computer interface[J]. arXiv preprint arXiv:2105.02786, 2021.
    - URL: https://arxiv.org/abs/2105.02786
    - Related Project: https://github.com/yi-ding-cs/LGG

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import SEEDDataset
        from torcheeg.models import LGGNet
        from torcheeg import transforms
        from torcheeg.datasets.constants import SEED_GENERAL_REGION_LIST

        dataset = SEEDDataset(root_path='./Preprocessed_EEG',
                              offline_transform=transforms.Compose([
                                  transforms.MeanStdNormalize(),
                                  transforms.To2d()
                              ]),
                              online_transform=transforms.Compose([
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]))
        model = LGGNet(region_list=SEED_GENERAL_REGION_LIST, chunk_size=128, num_electrodes=32, hid_channels=32, num_classes=2)
        
        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    The current built-in :obj:`region_list` includs:
    - torcheeg.datasets.constants.emotion_recognition.deap.DEAP_GENERAL_REGION_LIST
    - torcheeg.datasets.constants.emotion_recognition.dreamer.DREAMER_GENERAL_REGION_LIST


    Args:
        region_list (list): The local graph structure defined according to the 10-20 system, where the electrodes are divided into different brain regions.
        in_channels (int): The feature dimension of each electrode. (default: :obj:`1`)
        num_electrodes (int): The number of electrodes. (default: :obj:`32`)
        chunk_size (int): Number of data points included in each EEG chunk. (default: :obj:`128`)
        sampling_rate (int): The sampling rate of the EEG signals, i.e., :math:`f_s` in the paper. (default: :obj:`128`)
        num_T (int): The number of multi-scale 1D temporal kernels in the dynamic temporal layer, i.e., :math:`T` kernels in the paper. (default: :obj:`64`)
        hid_channels (int): The number of hidden nodes in the first fully connected layer. (default: :obj:`32`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.5`)
        pool_kernel_size (int): The kernel size of pooling layers in the temporal blocks (default: :obj:`16`)
        pool_stride (int): The stride of pooling layers in the temporal blocks (default: :obj:`4`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self,
                 region_list,
                 in_channels: int = 1,
                 num_electrodes: int = 32,
                 chunk_size: int = 128,
                 sampling_rate: int = 128,
                 num_T: int = 64,
                 hid_channels: int = 32,
                 dropout: float = 0.5,
                 pool_kernel_size: int = 16,
                 pool_stride: int = 4,
                 num_classes: int = 2):
        super(LGGNet, self).__init__()
        self.region_list = region_list
        self.inception_window = [0.5, 0.25, 0.125]

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.num_T = num_T
        self.hid_channels = hid_channels
        self.dropout = dropout
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes

        self.t_block1 = self.temporal_block(
            self.in_channels, self.num_T,
            (1, int(self.inception_window[0] * self.sampling_rate)),
            self.pool_kernel_size, self.pool_stride)
        self.t_block2 = self.temporal_block(
            self.in_channels, self.num_T,
            (1, int(self.inception_window[1] * self.sampling_rate)),
            self.pool_kernel_size, self.pool_stride)
        self.t_block3 = self.temporal_block(
            self.in_channels, self.num_T,
            (1, int(self.inception_window[2] * self.sampling_rate)),
            self.pool_kernel_size, self.pool_stride)

        self.bn_t1 = nn.BatchNorm2d(self.num_T)
        self.bn_t2 = nn.BatchNorm2d(self.num_T)

        self.cbam = CBAMBlock(num_electrodes)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(), nn.AvgPool2d((1, 2)))

        self.avg_pool = nn.AvgPool2d((1, 2))

        feature_dim = self.feature_dim
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(
            self.num_electrodes, feature_dim),
                                                requires_grad=True)
        self.local_filter_bias = nn.Parameter(torch.zeros(
            (1, self.num_electrodes, 1), dtype=torch.float32),
                                              requires_grad=True)

        self.aggregate = Aggregator(self.region_list)
        num_region = len(self.region_list)

        self.global_adj = nn.Parameter(torch.FloatTensor(
            num_region, num_region),
                                       requires_grad=True)

        self.bn_g1 = nn.BatchNorm1d(num_region)
        self.bn_g2 = nn.BatchNorm1d(num_region)

        self.gcn = GraphConvolution(feature_dim, hid_channels)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(num_region * hid_channels), num_classes))

        nn.init.xavier_uniform_(self.local_filter_weight)
        nn.init.xavier_uniform_(self.global_adj)

    def temporal_block(self, in_channels, out_channels, kernel_size,
                       pool_kernel_size, pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=(1, 1)),
            PowerLayer(kernel_size=pool_kernel_size, stride=pool_stride))

    def forward(self, x):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 1, 32, 128]`. Here, :obj:`n` corresponds to the batch size, :obj:`32` corresponds to :obj:`num_electrodes`, and :obj:`chunk_size` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        t1 = self.t_block1(x)
        t2 = self.t_block2(x)
        t3 = self.t_block3(x)
        x = torch.cat((t1, t2, t3), dim=-1)

        x = self.bn_t1(x)

        x = x.permute(0, 2, 1, 3)
        x = self.cbam(x)
        x = self.avg_pool(x)

        x = x.flatten(start_dim=2)
        x = self.local_filter(x)
        x = self.aggregate.forward(x)
        adj = self.get_adj(x)
        x = self.bn_g1(x)
        
        x = self.gcn(x, adj)
        x = self.bn_g2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    @property
    def feature_dim(self):
        mock_eeg = torch.randn(
            (1, self.in_channels, self.num_electrodes, self.chunk_size))

        t1 = self.t_block1(mock_eeg)
        t2 = self.t_block2(mock_eeg)
        t3 = self.t_block3(mock_eeg)
        mock_eeg = torch.cat((t1, t2, t3), dim=-1)

        mock_eeg = self.bn_t1(mock_eeg)
        mock_eeg = self.conv1x1(mock_eeg)
        mock_eeg = self.bn_t2(mock_eeg)
        mock_eeg = mock_eeg.permute(0, 2, 1, 3)
        mock_eeg = mock_eeg.flatten(start_dim=2)
        return mock_eeg.shape[-1]

    def local_filter(self, x):
        w = self.local_filter_weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        adj = torch.bmm(x, x.permute(0, 2, 1))
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(x.device)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    device = 'cpu'
    x = torch.randn(100,32,5)
    y = torch.randint(0, 2, (100, 1))

    print(x.shape)
    print(x.shape)
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size= 10, shuffle= True)


    model = DGCNN(in_channels= 5,
                  num_electrodes= 32,
                  num_layers= 2,
                  hid_channels= 32,
                  num_classes= 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    print(model(x).shape)

    # loop over the dataset multiple times
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Convert labels to Long type (for CrossEntropyLoss)
            labels = labels.squeeze().long()
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        print('Loss: {}'.format(running_loss))
    
    print('Finished Training')


