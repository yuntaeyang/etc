import pandas as pd
import numpy as np

path = "/content/drive/MyDrive/문장 유형 분류 AI 경진대회/open/"
train = pd.read_csv(f'{path}train.csv')
train.drop(columns=['ID'], inplace=True)
test = pd.read_csv(f'{path}test.csv')
test.drop(columns=['ID'], inplace=True)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
