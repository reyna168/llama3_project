import pandas as pd

# 讀取 CSV 檔案，並去除每個欄位中的空格和斷行符號
df = pd.read_csv('yilrobotwebv1.csv')

# 去除空格和斷行符號
df = df.apply(lambda x: x.str.replace(r'\s+', '', regex=True) if x.dtype == "object" else x)

# 將處理後的資料另存為新的 CSV 檔案
df.to_csv('output.csv', index=False)
