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
from models.simple_utils import unsupervised_train, mean, std, get_hms, save_checkpoint, invert, spare_visual





def main():
    batch = 512
    epochs = 200
    lr = 0.1

    def get_trainset():
        transform_train = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar10'], std['cifar10']),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        in_shape = [3, 32, 32]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
        return trainset, in_shape, trainloader

    trainset, in_shape, trainloader = get_trainset()

    # 参数定义区 --nBlocks 18 18 18 --nStrides 1 2 2 --nChannels 16 64 256

    # 初始下采样设置为2,初始扩充为12通道从3*32*32到12*16*16
    init_ds = 2
    # 定义各块层数，步幅（注意这里是仅在2处发生一次步幅为2），各层通道数
    nBlocks = [18, 18, 18]
    nStrides = [1, 2, 2]
    # 通道数尤其注意，设置为3 * init_ds^2 / 2以避免补0
    nChannels = [6, 24, 96]
    # 瓶颈乘数为4，这玩意限制了block中间卷积层的通道数，缩小4倍，经典的瓶颈层
    # 受限于输入图像的宽度，这里临时修改mult为2进行实验，原本值为4
    bottleneck_mult = 4


    def get_model():
        model = iRevNet(nBlocks=nBlocks, nStrides=nStrides,
                        nChannels=nChannels,init_ds=init_ds,
                        dropout_rate=0., affineBN=True,
                        in_shape=in_shape, mult=bottleneck_mult)
        fname = 'i-revnet-' + str(sum(nBlocks) + 1)# 每个Block不是有三个conv吗，命名不乘个3吗
        return model, fname

    model, fname = get_model()
    # 使用GPU
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=(0,))  # range(torch.cuda.device_count()))
    cudnn.benchmark = True

    # resume from cheakpoint
    is_resume = 0
    resume = 'checkpoint/Spare/i-revnet-55.t7'
    start_epoch = 1
    if is_resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    # 重建输入
    is_invert = 0
    if is_invert:
        if is_resume:
            invert(model, trainloader)
            return
        else:
            print("no model ready for invert, please use resume")

    # 将特征输出为卷积矩阵稀疏化的形式？存疑,这个后面可能要和invert整合起来
    is_sparse_visual = 0
    if is_sparse_visual:
        if is_resume:
            sparse_visual(model, trainloader)
            return
        else:
            print("no model ready for sparse visual, please use resume")

    print('|  Train Epochs: ' + str(epochs))
    print('|  Initial Learning Rate: ' + str(lr))

    elapsed_time = 0
    best_sparse_rate = 0.
    for epoch in range(start_epoch, 1+epochs):
        start_time = time.time()

        unsupervised_train(model, trainloader, trainset, epoch, epochs, batch, lr)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            })
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))


if __name__ == '__main__':
    main()