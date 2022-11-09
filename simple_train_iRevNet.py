'''
使用simple_i-RevNet, simple_utils简易搭一个无监督出来，先不用argparse

'''

import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time

from models.simple_iRevNet import iRevNet
from models.simple_utils import unsupervised_train, mean, std, get_hms





def main():
    def get_trainset():
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar10'], std['cifar10']),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        in_shape = [3, 32, 32]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        return trainset, in_shape, trainloader

    trainset, in_shape, trainloader = get_trainset()

    # 参数定义区 --nBlocks 18 18 18 --nStrides 1 2 2 --nChannels 16 64 256
    # 定义各块层数，步幅（注意这里是仅在2处发生一次步幅为2），各层通道数
    nBlocks = [18, 18, 18]
    nStrides = [1, 2, 2]
    nChannels = [16, 64, 256]
    # 初始下采样设置为0
    init_ds = 0
    # 瓶颈乘数为4，这玩意限制了block中间卷积层的通道数，缩小4倍，经典的瓶颈层
    bottleneck_mult = 4
    #
    epochs = 30
    lr = 0.1
    batch = 128
    def get_model():
        model = iRevNet(nBlocks=nBlocks, nStrides=nStrides,
                        nChannels=nChannels,init_ds=init_ds,
                        dropout_rate=0.1, affineBN=True,
                        in_shape=in_shape, mult=bottleneck_mult)
        fname = 'i-revnet-' + str(sum(nBlocks) + 1)# 每个Block不是有三个conv吗，命名不乘个3吗
        return model, fname

    model, fname = get_model()
    # 使用GPU
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=(0,))  # range(torch.cuda.device_count()))
    cudnn.benchmark = True



    print('|  Train Epochs: ' + str(epochs))
    print('|  Initial Learning Rate: ' + str(lr))

    elapsed_time = 0
    best_sparse_rate = 0.
    for epoch in range(1, 1+epochs):
        start_time = time.time()

        unsupervised_train(model, trainloader, trainset, epoch, epochs, batch, lr)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))


if __name__ == '__main__':
    main()