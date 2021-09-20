from os import path
import numpy as np
import pandas as pd

def csv_cal(path):
    df = pd.read_csv(path)