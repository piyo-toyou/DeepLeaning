import numpy as np
import pandas as pd

def excel_get(path, header=0, sheet_name="Sheet1"):
    df = pd.read_excel(path, header=header, sheet_name=sheet_name)
    gwl = df["GWL_m"].values
    rwl = df["RWL_m"].values
    rain = df["Rain_mm"].values
    return gwl, rwl, rain

def rwl_cal(rwl, n):
    rwl_dict = {}
    for i in range(1,n+1):
         rwl_dict[i] = rwl

def rain_cal():
    pass

my_xlsx = "Z://(新)技術課/CG20-0155-41 山財ダム/観測データ/時系列データとりまとめ.xlsx"
gwl, rwl, rain = excel_get(my_xlsx, header=0, sheet_name="EL")
