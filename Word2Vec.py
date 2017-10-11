import pandas as pd
import re

df = pd.read_csv("truyenkieu.txt",sep="/",names=["row"]).dropna()
print(df.head(10))

def transform_row(row):
    row = re.sub(r"^[0-9\.]+","",row)
    row = re.sub(r"[\.,\?]")