import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import models
from torch.autograd import Variable

from yolo_loss import yoloLoss
from dataset import YoloDataset
from yolo import Yolo


use_gpu = torch.cuda.is_available()
file_root = r'E:\datasets\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
learning_rate = 0.001
num_epochs = 100
batch_size = 64


if __name__ == '__main__':
    net = Yolo()

    print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

    criterion = yoloLoss(7, 2, 5, 0.5)
    if use_gpu:
        net.cuda()

    net.train()
    params = net.parameters()
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

    train_dataset = YoloDataset(root=file_root, list_file='voc2012.txt', train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # test_dataset = YoloDataset(root=file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
    # test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))
    logfile = open('log.txt', 'w')

    num_iter = 0
    best_test_loss = np.inf
    for epoch in range(num_epochs):
        net.train()
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.

        for i, (images, target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            loss = criterion(pred, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
                num_iter += 1
        
        if (epoch + 1) % 10 == 0:
            # 选择一个文件名或路径来保存模型，这里使用epoch数作为文件名的一部分
            model_save_path = 'model_epoch_{}.pth'.format(epoch + 1)
            # 保存模型的状态字典
            torch.save(net.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')