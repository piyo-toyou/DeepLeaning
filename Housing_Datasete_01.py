import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

class Housing(Dataset):
    def __init__(self, csv_path):
        # csv ファイルを読み込む。
        df = pd.read_csv(csv_path, sep=",", header=0)
        x = df.iloc[:, 0:-1]
        t = df.iloc[:, -1]
        data = torch.tensor(x.values, dtype=torch.float32)  # データ (2 ~ 14列目)
        labels = torch.tensor(t.values, dtype=torch.float32)  # ラベル (1列目)
        # データを標準化する。
        # data = torch.from_numpy(normalize(data)).float()

        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """サンプルを返す。
        """
        return self.data[index], self.labels[index]

    def __len__(self):
        """csv の行数を返す。
        """
        return len(self.data)