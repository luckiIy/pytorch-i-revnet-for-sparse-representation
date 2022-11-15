'''
iRevNet.py中作者为了实现双射对原函数做的修改已经严重的影响到了我的代码阅读，这里重取仅可逆部分的方便理解
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_utils import split, merge, psi, injective_pad

# 构成iRevnet的基本块
# test3,修改block使得最后一层输出变为ReLU而非Conv直觉上这样会增加得0率
class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.pad = 2 * out_ch - in_ch  # pad为啥这样算？
        self.stride = stride

        self.inj_pad = injective_pad(self.pad)  # 是因为这里是单补0吗

        self.psi = psi(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print('')
            print('| Injective iRevNet |')  # SOGA，用来控制论文里介绍的那个单射网络的
            print('')
        layers = []
        # if not first:

        layers.append(nn.Conv2d(in_ch // 2, int(out_ch // mult), kernel_size=3,
                                stride=stride, padding=1, bias=False))  # 这里来回乘2除2好像都是为了那个单射的网络匹配？真是严重降低了代码可读性
        layers.append(nn.BatchNorm2d(int(out_ch // mult), affine=affineBN))  # affineBN=1即保证归一化中gama和beta可学习
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch // mult), int(out_ch // mult),
                                kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm2d(int(out_ch // mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch // mult), out_ch, kernel_size=3,
                                padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))

        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective block forward 这里很明显是那个保留一部分的结构"""
        # 这里很重要，目前还没理解，通过这里之后通道数从1，2变成了16，16大量在这里补0
        if self.pad != 0 and self.stride == 1:  # 这里的pad是那个大量补0的pad吧。。。injective就会进去
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x) #先合并，然后做一个forward，再合在一起
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        # 真正进入网络的地方，所以这里经常出问题，断点设在这
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1   # Y1 = X1 + F(X2(Y2))
        return (x2, y1)

    def inverse(self, x):
        """ bijective block inverse 求原输入的逆过程"""
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1   # X1 = Y1 - F（Y2（X2））
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x

# i-RevNet
class iRevNet(nn.Module):
    def __init__(self, nBlocks, nStrides, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        '''
        in_shape 是单个输入图像的shape (3, n_H, n_W)
        '''
        super(iRevNet, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)# 不懂
        self.init_ds = init_ds
        # 这里初始化第一层in_ch 输入通道数（3）* 2^init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))   # 每个Block有3层，最后一个1还没看到在哪
        if not nChannels:   # nChannels大概是约束了通道数的变化，就是那个利用特殊结构扩充通道数，每次是扩充4倍
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds) # psi一直不理解，注意这里是用ds初始化的
        # 事实上这里输入的就是nChannal啊，完全由之前那个矩阵控制

        # 更好的做法为进行初始下采样
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        # # 下面这两行应该是最后了？，把最后特征的通道*2，BN后线性到类上
        # self.bn1 = nn.BatchNorm2d(nChannels[-1]*2, momentum=0.9)
        # self.linear = nn.Linear(nChannels[-1]*2, nClasses)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        # 对应的四次尺度变化,得到各层的对应stride和channel，注意stride仅在采样层为2，而channel在对应位置变为原值的4倍
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            # in_ch = 2 * channel自始至终我都没明白这个2*是想干嘛，是方便那个单射？等下试试能不能都改掉
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        n = self.in_ch//2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            # 这里block返回的事实上就是block类的单个block,沿list不断进行前向
            out = block.forward(out)
        # 得到前向输出，其中out_bij储存了可用于逆的特征
        out_bij = merge(out[0], out[1])
        # # BN, ReLU
        # out = F.relu(self.bn1(out_bij))
        # # 平均值池化，ds是池化的kernel size？但是这里的意图在于计算最后输出特征的n_H,n_W,直接对单个二维图平均值作为一维特征输出，就是不知道咋算的
        # out = F.avg_pool2d(out, self.ds)
        # # 整理特征，送到线性映射，得到最终的输出out
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # return out, out_bij
        return out_bij

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = split(out_bij)
        for i in range(len(self.stack)):
            # 方向逐层inverse
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0],out[1])
        # 这里psi是干嘛的还是不懂
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out
        return x
if __name__ == '__main__':
    # nBlocks 对应按块分割的各块中有多少层
    # nStrides 还不是很懂
    # nChannels=[24, 96, 384, 1536]变换后通道数
    model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                    nChannels=[24, 96, 384, 1536], nClasses=1000, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[3, 224, 224],
                    mult=4)
    y = model(Variable(torch.randn(1, 3, 224, 224)))
    print(len(y))
    # nBlocks = [6, 16, 72, 6]
    # nStrides = [2, 2, 2, 2]
    # nChannels = [24, 96, 384, 1536]
    # strides = []
    # channels = []
    # for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
    #     strides = strides + ([stride] + [1] * (depth - 1))
    #     channels = channels + ([channel] * depth)
    # print(strides)
    # print(channels)
    # print([1])
    print("end")
