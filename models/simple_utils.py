'''
按着utils_cifar写的
'''
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable


import os
import sys
import math
import numpy as np
# test1
# 经测试，用L1Loss作为损失，LOSS会逐渐降低，输出特征会逐渐减小，但就是不会归0。。这也不是不收敛啊，不收敛指的应该是LOSS没法稳定下降，所以接下来试试smoothL1不行再看看是不是结构出问题了
# criterion = nn.L1Loss()

# test2
criterion = nn.SmoothL1Loss()


# 各通道normal的均值和标准差，这组数据不知道是咋来的，以及为啥要normal，大概是跑了一次之后指数平均？
mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# test2.0中仅改变Loss无效，结果是收敛确实变快了，但特征就是不为0？不知道是不是pytorch有什么特别的正则化机制？还是受BN影响？
# 接下来尝试逐步降低lr，之前一直是小epoch所以lr一直为0.1，但考虑到收敛速度很快，在10，20，30分别降低lr
# 看看是否是出现了那种来回交替就是不归0的情况，但如果真的是这样感觉大概率是不可行的，但看看LOSS会不会大幅度下降
def learning_rate(init, epoch):
    # 随epoch增加到一定程度降低lr
    optim_factor = 0
    if(epoch > 30):
        optim_factor = 3
    elif(epoch > 20):
        optim_factor = 2
    elif(epoch > 10):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def unsupervised_train(model, trainloader, trainset, epoch, epoch_n, batch_size, lr):
    model.train()
    # 按epoch统计稀疏率
    zero_num = 0
    total = 0
    loss_sum = 0
    # 这里是SGD优化，但是超参数的选取不知道是怎么来的，主要是后面这个decay
    optimizer = optim.SGD(model.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)

    # 这里是得到并输出参数总量params
    # 这里是个技巧，虽然有点复杂
    # 首先是lambda定义了一个对p的任意函数，而model.parameters是model中所有tensor参数的集合，函数lambda中的.requires_grad属性保证了过滤器得到的是可变参数
    # 即model_parameters包含了model中所有的可变参数，这个filter结构也被用来固定部分参数训练另一部分的挑选
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('|  Number of Trainable Parameters: ' + str(params))

    # 输出epoch和lr
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, learning_rate(lr, epoch)))

    # 在batch上循环
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        out = model(inputs)
        zero_tensor = torch.zeros(out.shape)
        zero_tensor = zero_tensor.cuda()
        zero_tensor = Variable(zero_tensor)
        loss = criterion(out, zero_tensor)
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
        # 修正loss的tensor格式，虽然我也不知道这样做相比于直接loss.data = torch.reshape(loss.data, (1,))的意义（测试时都会进去）但好像这样更正规
        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        loss_sum += loss.data[0]
        zero_num += out.eq(zero_tensor).cpu().sum()
        total += np.prod(out.size())
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, epoch_n, batch_idx + 1,
                            (len(trainset) // batch_size) + 1, loss_sum / batch_idx, 100.*zero_num/total))
        sys.stdout.flush()


# test部分暂时先不写吧，先去试试train了
def test(model):
    model.eval()