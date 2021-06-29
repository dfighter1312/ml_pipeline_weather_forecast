import numpy as np
import pandas as pd
import json
import os


class Standardization():

    def __init__(self, __C):
        self.__C = __C
        self.file_name = 'std_' + __C.DATA_CLASS + '.json'

        if self.__C.RUN_MODE == 'test':
            with open(os.path.join(self.__C.CKPTS_PATH, self.file_name), 'r') as fp:
                saved_dict = json.loads(fp.read())
                self.mean = pd.Series(saved_dict['mean'])
                self.std = pd.Series(saved_dict['std'])

    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()

        # Save mean and standard into a json file
        save_dict = {
            'version': 0,
            'mean': self.mean.to_dict(),
            'std': self.std.to_dict(),
        }

        with open(os.path.join(self.__C.CKPTS_PATH, self.file_name), 'w') as fp:
            json.dump(save_dict, fp)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X, label_columns=None):
        if label_columns is None:
            label_columns = self.mean.columns
        return X * self.std[label_columns].to_numpy() + self.mean[label_columns].to_numpy()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
