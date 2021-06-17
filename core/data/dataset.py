from pandas import read_csv
from core.utils.preprocess import processing

class Dataset():

    def __init__(self, __C):
        self.__C = __C
        self.df = read_csv(__C.DATA_PATH[__C.RUN_MODE], encoding="ISO-8859-1", 
                            index_col='Date Time', parse_dates=['Date Time'], 
                            dayfirst=True)

    def preprocess(self):
        self.df = processing(self.df)