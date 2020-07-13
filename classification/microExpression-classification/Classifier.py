import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        # super继承父类
        super(Classifier, self).__init__()
        
        # conv2d( in_channels, out_channels, kernel_size, stide, padding)
        # input shape: [1, 48, 48]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 2, 1, 1), # [64, 49, 49]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[64, 25, 25]
            
            nn.Conv2d(64, 128, 2, 1, 1), # [128, 26, 26]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), # [128, 13, 13]
            
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 13, 13]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [256, 7, 7]
            
            nn.Conv2d(256, 256, 3, 1, 1), #[256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[256, 4, 4]        
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, 7)
        )
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
