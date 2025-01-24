import torch.nn as nn


class CBL(nn.Module):
    """
    Conv-BN-LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False) # 使用了BN，不需要偏置
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)  # 原地操作数据，减少内存开销，提升速度

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x
    

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.backbone = nn.Sequential(
            # 输入 (448, 448, 3)
            CBL(3, 64, 7, 2, 3),  # (224, 224, 64)
            nn.MaxPool2d(2, 2),  # (112, 112, 64)
            
            CBL(64, 192, 3, 1, 1),   # (112, 112, 192)
            nn.MaxPool2d(2, 2),  # (56, 56, 192)
            
            CBL(192, 128, 1, 1, 0), # (56, 56, 128)
            CBL(128, 256, 3, 1, 1), # (56, 56, 256)
            CBL(256, 256, 1, 1, 0), # (56, 56, 256)
            CBL(256, 512, 3, 1, 1), # (56, 56, 512)
            nn.MaxPool2d(2, 2), # (28, 28, 512)
            
            CBL(512, 256, 1, 1, 0), # (28, 28, 256)
            CBL(256, 512, 3, 1, 1), # (28, 28, 512) ,重复4次
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1), # (28, 28, 512)
            CBL(512, 512, 1, 1, 0), # (28, 28, 512)
            CBL(512, 1024, 3, 1, 1), # (28, 28, 1024)
            nn.MaxPool2d(2, 2), # (14, 14, 1024)

            CBL(1024, 512, 1, 1, 0),  # 重复两次
            CBL(512, 1024, 3, 1, 1), 
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1), # (14, 14, 1024)
            CBL(1024, 1024, 3, 1, 1), # (14, 14, 1024)
            CBL(1024, 1024, 3, 2, 1), # (7, 7, 1024)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class Head(nn.Module):
    """
    检测头由两个全连接层构成，第一层 7x7x1024->4096, 第二层4096->7x7*30
    """
    def __init__(self, num_classes=20):
        super(Head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(7*7*1024, 1024, bias=True),  # 这里偏置不能少
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, (num_classes+10)*7*7, bias=True),  # 修改了4096->1024
        )
        
    def forward(self, x):
        return self.classifier(x)
