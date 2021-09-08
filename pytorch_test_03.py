import torch
import numpy as np
import pandas as pd

t1 = torch.ones(4, 4)
t2 = torch.cat([t1, t1, t1], dim=1)
t1[:, 1] = 0
print(t2)

t1.add_(5)
