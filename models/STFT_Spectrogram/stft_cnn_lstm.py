import torch
import torch.nn as nn

class STFT_LSTM_CNN_Model(nn.Module):
    def __init__(self,in_channel=14, hidden_size1=64, hidden_size2= 128, hidden_size_lstm= 256, num_layers= 1, num_classes=1, dropout_prob=0.5):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_prob)
        self.af = nn.ReLU()
        
        # CNN
        self.conv1 = nn.Conv2d(in_channels= in_channel, out_channels=hidden_size1 , kernel_size= (3,1) ,padding=2)
        self.bn1 = nn.BatchNorm2d(num_features= hidden_size1)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels= hidden_size1, out_channels=hidden_size2 , kernel_size= 3 ,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features= hidden_size2)
        self.max_pool2 = nn.MaxPool2d(4)

        # LSTM
        self.num_layers = num_layers
        self.hidden_size_lstm = hidden_size_lstm
        self.lstm = nn.LSTM(hidden_size2, hidden_size_lstm, num_layers, batch_first=True)


        # fully connected
        self.fc = nn.LazyLinear(num_classes)


    def forward(self, x): # [batch, 14, 128]
        x = self.conv1(x) # [batch, 64, 64]
        x = self.bn1(x)
        x = self.af(x)
        x = self.dropout(x)
        x = self.max_pool1(x) # [batch, 64, 32]

        x = self.max_pool2(self.dropout(self.af(self.bn2(self.conv2(x))))) # [batch, 128, 16] -> F x L


        # lstm section
        x = x.squeeze(-1)       
        x = x.permute(0, 2, 1) # (Batch, Length, features = 128 channels)

        # Initialize hidden state and cell state with random values
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size_lstm).to(x.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size_lstm).to(x.device)

        # Forward propagate the LSTM
        out, (_,_) = self.lstm(x, (h0, c0)) #

        # x = self.fc(out[:, -1, :])
        x = self.fc(out.flatten(1))
        x = torch.sigmoid(x)

        return x



if __name__ == "__main__":
    model = STFT_LSTM_CNN_Model()
    
    data = torch.randn(10,14,33,5) # (batch, channel, height, width)
    target = torch.randn(1,1)
    # dataset = TensorDataset(data, target)
    # dataloader = DataLoader(dataset, batch_size= 10, shuffle= True)
    # print(len(dataloader))

    print(model(data).shape)
    
