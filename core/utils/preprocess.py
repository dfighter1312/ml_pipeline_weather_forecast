import numpy as np
import pandas as pd
import datetime


def processing(df):
    """
    Perform preprocessing data for data preparation.
    (Data cleaning and feature engineering).
    """

    # Replace errornous data with forward fill
    df.replace(-9999, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Change wind vector from wv, max. wv and wd
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv*np.cos(wd_rad)
    df['max Wy'] = max_wv*np.sin(wd_rad)

    # Create day/year sin/cos columns as signals for Date Time
    timestamp = df.index.map(datetime.datetime.timestamp)
    day = 24*60*60
    year = (365.2425)*day
    df['Day sin'] = np.sin(timestamp * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

    # Drop redundant columns
    drop_columns = ['Tpot (K)', 'H2OC (mmol/mol)', 'SWDR (W/m²)']
    df.drop(columns=drop_columns, inplace=True)

    # Use binning for rain column
    rain = df.pop('rain (mm)')
    raining = df.pop('raining (s)')
    df['No rain'] = rain == 0
    df['Light rain'] = (rain < (2.5 / 6)) & (rain > 0)
    df['Moderate/Heavy/Violent rain'] = (rain >= (2.5 / 6))
    
    return df