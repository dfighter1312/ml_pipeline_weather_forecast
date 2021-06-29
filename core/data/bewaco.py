import datetime
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.data.base_dataset import BaseDataset


class BewacoDataset(BaseDataset):

    def __init__(self, __C):
        self.__C = __C
        self.create_dataframe()
        self.preprocess()

    def create_dataframe(self):
        column_names = [
            'Date Time',
            'S1-Batt Volt',
            'S1-PTemp',
            'S1-Salt',
            'S1-SpCond',
            'S1-Temp',
            'S2-Batt Volt',
            'S2-PTemp',
            'S2-Salt',
            'S2-SPCond',
            'S2-Temp',
            'S2-Wiper cur',
            'S2-Wiper pos',
            'S3-Batt Volt',
            'S3-PTemp',
            'S3-Salt'
        ]
        self.df = pd.read_csv(self.__C.DATA_PATH['bewaco'][self.__C.RUN_MODE], names=column_names,
                              header=2, parse_dates=['Date Time'])

    def get_columns(self):
        return self.df.columns

    def get_str_label_columns(self):
        label_columns = self.__C.LABEL_COLUMNS

        try:
            for i, ele in enumerate(self.__C.LABEL_COLUMNS):
                if isinstance(ele, int):
                    label_columns[i] = self.get_columns()[i]
        except IndexError:
            print('You should check the LABEL_COLUMN input')

        self.__C.LABEL_COLUMNS = label_columns

        return label_columns


    def preprocess(self):
        # Drop the first row
        self.df.drop(index=0, inplace=True)

        # Replace erronous value to missing value
        self.df.replace(-99.99, np.nan, inplace=True)
        error_s1temp = self.df['S1-Temp'] < 1
        error_s2temp = self.df['S2-Temp'] < 1
        self.df.loc[error_s1temp, 'S1-Temp'] = np.nan
        self.df.loc[error_s2temp, 'S2-Temp'] = np.nan

        # Change into 30-minute data
        thirty_minutes = 60 * 30
        selected_rows = self.df['Date Time'].map(datetime.timestamp) % thirty_minutes == 0
        self.df = self.df[selected_rows]
        self.df.reset_index(drop=True, inplace=True)

        # Check if every rows is recorded after 30 minutes
        missing_times = []
        for i in range(self.df.shape[0] - 1):
            if (self.df.loc[i+1, 'Date Time'] - self.df.loc[i, 'Date Time']) != timedelta(minutes=30):
                missing_times.append((self.df.loc[i, 'Date Time'], self.df.loc[i+1, 'Date Time']))

        df_all_missing_timestamps = pd.DataFrame(columns=self.df.columns)
        for f, t in missing_times:
            i = f + timedelta(minutes=30)
            while i < t:
                df_new = pd.DataFrame({'Date Time': i}, index=[0], columns=self.df.columns)
                df_all_missing_timestamps = df_all_missing_timestamps.append(df_new)
                i += timedelta(minutes=30)
        
        self.df = self.df.append(df_all_missing_timestamps).sort_values('Date Time')

        # Sum 2 S2-Wiper columns
        self.df['S2-Wiper sum'] = self.df.pop('S2-Wiper cur') + self.df.pop('S2-Wiper pos')

        # Fill missing values
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')

        # Replace outliers
        self.df.set_index('Date Time', inplace=True)
        for col in self.df.columns:
            self.df = self._replace_outliers(self.df, col)

        # Create day/year sin/cos columns as signals for Date Time
        timestamp = self.df.index.map(datetime.timestamp)
        day = 24*60*60
        year = (365.2425)*day
        self.df['Day sin'] = np.sin(timestamp * (2 * np.pi / day))
        self.df['Day cos'] = np.cos(timestamp * (2 * np.pi / day))
        self.df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
        self.df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))


    def set_predict_dataset(self):
        if self.df.shape[0] < self.__C.N_HISTORY_DATA:
            raise Exception(
                f"""The provided dataset must have number of records greater or
                equal to {self.__C.N_HISTORY_DATA}""")
        # When predict data has more row than history data, get the last N_HISTORY_DATA
        self.df = self.df[-self.__C.N_HISTORY_DATA:]

    def _replace_outliers(self, data, col):
        lower_range = data[col].quantile(0.10)
        upper_range = data[col].quantile(0.90)
        data[col] = np.where((data[col] < lower_range), lower_range, data[col])
        data[col] = np.where((data[col] > upper_range), upper_range, data[col])
        
        return data