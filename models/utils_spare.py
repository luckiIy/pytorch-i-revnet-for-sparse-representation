'''
这里主要用于存储稀疏性，低秩性计算相关函数
'''

import torch
import torch.nn as nn

import os
import sys
import math
import numpy as np
import time

# 定义基尼系数的计算方式
# 论文里确定矩阵的低秩性用的是奇异值的GINI，这里后面可以详细看看，说不定要改
def compute_gini(feature):
    # 输入feature应该是提取到的特征，tensor型数据,转为numpy，转成向量，获得长度，取绝对值变为标准的可measure非负序列
    feature = feature.cpu().detach().numpy()
    feature = feature.reshape(-1)
    n = feature.shape[0]

    # 利用compare measure那篇文章的公式，求1范数，排序，求解gini
    feature = np.abs(feature)
    f_sum = np.sum(feature)
    feature = np.sort(feature)
    k = np.arange(n) + 1.
    gini =1. - 2. * np.sum((n + 0.5 - k) / (f_sum * n) * feature)
    return gini

# 一维circshift
def circshift(x, k):
    # 目前先做一维的, X是原输入, np数组, k是移动步长, k为正时行向量循环右移
    x = x.reshape(-1)
    return np.concatenate((x[-k:], x[:-k]))

# 特征展开为向量得到卷积矩阵,k取-1则按卷积矩阵为方阵输出
def conv_matrix(feature,k=-1):
    # 输入feature应该是提取到的特征，tensor型数据,转为numpy，转成向量，获得长度，取绝对值变为标准的可measure非负序列
    feature = feature.cpu().detach().numpy()
    feature = feature.reshape(-1)
    n = feature.shape[0]
    if k == -1:
        k = n
    conv_m = np.zeros((n,k))
    for i in range(k):
        conv_m[:, i] = circshift(feature, i)
    return conv_m
# feature = torch.randn(2, 3, 2)
# conv_m = conv_matrix(feature,3)
# print(feature,conv_m)