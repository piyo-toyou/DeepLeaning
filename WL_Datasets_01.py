import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

class WaterLevel(Dataset):
    def __init__(self, csv_path):
        # csv ファイルを読み込む。
        df = pd.read_csv(csv_path)
        data = df.iloc[:, 1:]  # データ (2 ~ 14列目)
        values = df.iloc[:, 0]  # ラベル (1列目)
        # データを標準化する。
        data = normalize(data)

        self.data = data
        self.values = values

    def __getitem__(self, index):
        """サンプルを返す。
        """
        return self.data[index], self.values[index]

    def __len__(self):
        """csv の行数を返す。
        """
        return len(self.data)