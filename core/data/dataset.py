import datetime
import pandas as pd
import numpy as np


class JenaDataset():

    def __init__(self, __C):
        self.__C = __C
        self.df = pd.read_csv(__C.DATA_PATH[__C.RUN_MODE], encoding="ISO-8859-1",
                           index_col='Date Time', parse_dates=['Date Time'],
                           dayfirst=True)
        self.preprocess()

    def preprocess(self):
        """
        Perform preprocessing data for data preparation.
        (Data cleaning and feature engineering).
        """

        # Replace errornous data with forward fill
        self.df.replace(-9999, np.nan, inplace=True)
        self.df.fillna(method='ffill', inplace=True)

        # Change wind vector from wv, max. wv and wd
        wv = self.df.pop('wv (m/s)')
        max_wv = self.df.pop('max. wv (m/s)')

        # Convert to radians.
        wd_rad = self.df.pop('wd (deg)') * np.pi / 180

        # Calculate the wind x and y components.
        self.df['Wx'] = wv * np.cos(wd_rad)
        self.df['Wy'] = wv * np.sin(wd_rad)

        # Calculate the max wind x and y components.
        self.df['max Wx'] = max_wv*np.cos(wd_rad)
        self.df['max Wy'] = max_wv*np.sin(wd_rad)

        # Create day/year sin/cos columns as signals for Date Time
        timestamp = self.df.index.map(datetime.datetime.timestamp)
        day = 24*60*60
        year = (365.2425)*day
        self.df['Day sin'] = np.sin(timestamp * (2 * np.pi / day))
        self.df['Day cos'] = np.cos(timestamp * (2 * np.pi / day))
        self.df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
        self.df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

        # Drop redundant columns
        drop_columns = ['Tpot (K)', 'H2OC (mmol/mol)', 'SWDR (W/m²)']
        self.df.drop(columns=drop_columns, inplace=True)

        # Use binning for rain column
        rain = self.df.pop('rain (mm)')
        # raining = self.df.pop('raining (s)')
        self.df['No rain'] = rain == 0
        self.df['Light rain'] = (rain < (2.5 / 6)) & (rain > 0)
        self.df['Moderate/Heavy/Violent rain'] = (rain >= (2.5 / 6))