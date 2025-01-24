import math
import torch
import torch.nn.functional as F
from model import *



class Yolo(nn.Module):
    """
    Yolo网络由backbone和head构成，backbone输出7x7x1024，head输出7x7x30
    """
    def __init__(self, num_classes=20):
        super(Yolo, self).__init__()
        self.backbone = Backbone()
        self.head = Head(num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.backbone(x)
        # batch_size * channel * width * height
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # 平铺向量
        x = self.head(x)
        x = F.sigmoid(x) # 归一化到0-1
        x = x.view(-1,7,7,30) # 重塑成bs,7,7,30张量
        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 448, 448))
    net = Yolo()
    print(net(x).shape)
