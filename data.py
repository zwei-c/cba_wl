from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
# 處理資料
# 資料集取自 uci machine learning repository
# 資料格式為.data [n_samples, n_features]，.target [n_samples]
# 為資料集進行one-hot encoding並取得屬性重要性


class Data():

    def __init__(self, dataName):

        self.dataPath = os.path.join(os.getcwd(), "data", dataName)
        self.data: pd.DataFrame = None
        self.encodedData: pd.DataFrame = None
        self.labelEncoder = LabelEncoder()
        self.randomForestClassifier = RandomForestClassifier()
        self.featureImportances: list = None

    def readData(self):
        self.data = pd.read_csv(self.dataPath, header=None)

    def encodeData(self):
        # 處理缺失值
        # 使用眾數填充類別型特徵的缺失值
        self.data = self.data.fillna(self.data.mode().iloc[0])
        # 使用中位數填充數值型特徵的缺失值
        self.data = self.data.fillna(self.data.median())

        # 處理數值型特徵
        # 將數值型特徵轉換為類別型特徵

        self.encodedData = self.data.apply(self.labelEncoder.fit_transform())
