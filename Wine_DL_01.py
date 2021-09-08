import Wine_Datasete_02 as wd2
import torch
from torch.utils.data import DataLoader

# Dataset を作成する。
dataset = wd2.Wine("https://git.io/JfodD")
# DataLoader を作成する。
dataloader = DataLoader(dataset, batch_size=64)
