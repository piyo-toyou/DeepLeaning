import torch
import numpy as np
import pandas as pd
from torch.functional import Tensor

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_data = np.array(data)
x_np = torch.from_numpy(np_data)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_ones)
print(x_rand)
