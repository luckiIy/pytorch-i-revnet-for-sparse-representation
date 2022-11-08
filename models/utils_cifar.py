# Author: J.-H. Jacobsen.
#
# Modified from Pytorch examples code.
# Original license shown below.
# =============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from models.model_utils import get_all_params

import os
import sys
import math
import numpy as np

criterion = nn.CrossEntropyLoss()

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def learning_rate(init, epoch):
    # 随epoch增加到一定程度降低lr
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def train(model, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, in_shape):
    '''
    epoch 是当前epoch
    num_epoch 是总数，为了方便显示
    in_shape 没用上？？？
    '''
    model.train()
    train_loss = 0
    correct = 0
    total = 0
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
    # 这里就是需要改动的地方了，事实上稀疏性作为约束优化的结果是没有targets的
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out, out_bij = model(inputs)               # Forward Propagation
        loss = criterion(out, targets)  # Loss 上面定义的交叉熵损失
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
        # 修正loss的tensor格式
        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        # 求当前epoch中loss的和
        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.data[0], 100.*correct/total))
        sys.stdout.flush()


def test(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, out_bij = model(inputs)
        loss = criterion(out, targets)

        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        test_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
                'model': model if use_cuda else model,
                'acc': acc,
                'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+fname+'.t7')
        best_acc = acc
    return best_acc
