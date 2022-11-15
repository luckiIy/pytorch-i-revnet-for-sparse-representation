import torch
import torch.nn as nn

import numpy as np

import time

# zero_tensor = torch.zeros(5)
# out = torch.tensor([0,1,1,3,0])
# zero_num = out.eq(zero_tensor).cpu().sum()/5
# print(zero_num)


# # 定义基尼系数的计算方式
# def compute_gini(feature):
#     # 输入feature应该是提取到的特征，tensor型数据,转为numpy，转成向量，获得长度，取绝对值变为标准的可measure非负序列
#     feature = feature.cpu().numpy()
#     feature = feature.reshape(-1)
#     n = feature.shape[0]
#
#     # 利用compare measure那篇文章的公式，求1范数，排序，求解gini
#     feature = np.abs(feature)
#     f_sum = np.sum(feature)
#     feature = np.sort(feature)
#     k = np.arange(n) + 1.
#     gini =1. - 2. * np.sum((n + 0.5 - k) / (f_sum * n) * feature)
#     return gini
#
#
# feature = torch.randn(2, 3, 2)
# start_time = time.time()
# gini = compute_gini(feature)
# time_gini = time.time() - start_time
#
# print('input feature:', feature)
# print('time: %.3f' % time_gini)
# print('Compute Gini:%.3f'% gini)

