import torch
import torch.nn as nn


zero_tensor = torch.zeros(5)
out = torch.tensor([0,1,1,3,0])
zero_num = out.eq(zero_tensor).cpu().sum()/5
print(zero_num)