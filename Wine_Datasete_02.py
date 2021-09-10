import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

class Wine(Dataset):
    def __init__(self, csv_path):
        # csv ファイルを読み込む。
        df = pd.read_csv(csv_path)
        data = np.array(df.iloc[:, 1:])  # データ (2 ~ 14列目)
        labels = np.array(df.iloc[:, 0], dtype="long")  # ラベル (1列目)
        # データを標準化する。
        data = normalize(data)
        # クラス ID を 0 始まりにする。[1, 2, 3] -> [0, 1, 2]
        labels -= 1

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