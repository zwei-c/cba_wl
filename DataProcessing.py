from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
# 處理資料
# 資料集取自 uci machine learning repository
# 資料格式為.data [n_samples, n_features]，.target [n_samples]
# 為資料集進行one-hot encoding並取得屬性重要性


class DataProcessing():

    def __init__(self, data_name):
        self.data_path = os.path.join(os.getcwd(), "data", data_name)
        self.data: pd.DataFrame = None
        self.encoded_data: pd.DataFrame = None
        self.feature_importances: list = None

    def read_data(self):
        self.data = pd.read_csv(self.data_path, header=None)

    def process(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = self.data.select_dtypes(
            exclude=[np.number]).columns

        for col in numeric_cols:  # fill missing values with median column values
            self.data[col].fillna(self.data[col].median(), inplace=True)

        for col in non_numeric_cols:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

        for col in numeric_cols:
            self.data[col] = pd.cut(self.data[col], 5, labels=False)

        label_encoder = LabelEncoder()
        self.encoded_data = self.data.apply(
            label_encoder.fit_transform)

        self.encoded_data.rename(
            {self.encoded_data.columns[-1]: "class"}, axis=1, inplace=True)

    def get_feature_importances(self):
        X = self.encoded_data.iloc[:, :-1]
        y = self.encoded_data.iloc[:, -1]
        model = RandomForestClassifier()
        model.fit(X, y)
        self.feature_importances = model.feature_importances_


if __name__ == "__main__":
    data_processing = DataProcessing("adult.data")
    data_processing.read_data()
    data_processing.process()
    print(data_processing.encoded_data)
    # data_processing.get_feature_importances()
    # print(data_processing.feature_importances)
