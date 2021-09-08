import torch
import numpy as np
import pandas as pd

t1 = torch.ones(4, 4)
t2 = torch.cat([t1, t1, t1], dim=1)

print(t2)

y1 = t1 @ t1.T
y2 = t1.matmul(t1.T)
y3 = torch.rand_like(t1)
print(y3)
torch.matmul(t1, t1.T, out=y3)
print(y3)

z1 = t1 * t1
print(z1)