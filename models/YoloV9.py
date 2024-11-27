# Using backbone of YoLo V9 instead of our feature extraction head
import torch
import torch.nn as nn




def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class ELAN1(nn.Module):

    def __init__(self, c1, c2, c3, c4):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3//2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
    


class FeatureExtractor(nn.Module):
        # def __init__(self, c1=14, c2=64, c3=128, c4=128):
        def __init__(self, c1=1, c2=64, c3=128, c4=128):
            super(FeatureExtractor, self).__init__()
            # Initialize the ELAN1 layer (using channels c1=14, c2=64, etc.)
            self.elan1 = ELAN1(c1, c2, c3, c4)

        def forward(self, x):
            # Forward pass through the ELAN1 block to extract features
            features = self.elan1(x)
            return features

class EmotionClassifier(nn.Module):
    def __init__(self, input_size=14, num_classes=2):
        super(EmotionClassifier, self).__init__()
        # self.fc = nn.Linear(input_size, num_classes)  # Fully connected layer for classification
        self.fc = nn.LazyLinear(num_classes)  # Fully connected layer for classification

    def forward(self, x):
        return self.fc(x.flatten(1))  # Flatten the output for classification


class YOLO9_Backbone_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = EmotionClassifier(num_classes=1) # 1 for binary
        self.sigmoid_fn = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        out = self.classifier(features)
        out = self.sigmoid_fn(out)
        
        return out



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    
    model = YOLO9_Backbone_Classifier()

    # Define a sample input based on your dataset format
    sample_input = torch.randn(1, 14, 128)  # Batch size 1, 14 channels, 128 timesteps (or spatial features)


    # Pass the sample input through the model to get feature output
    output = model(sample_input.unsqueeze(0))


    print(output.shape)  # Check the shape of extracted features
    print(output)  # Check the shape of extracted features
