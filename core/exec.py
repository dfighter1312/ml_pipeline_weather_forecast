import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from core.utils.train_test_split import train_test_split
from core.data.dataset import JenaDataset
from core.data.window import WindowGenerator
from core.utils.standardize import Standardization
from core.model.factory import Factory


class Execution():
    """Execution class. Part of the entrypoint."""

    def __init__(self, __C):
        self.__C = __C
        self.dataset = JenaDataset(self.__C)
        if __C.wandb:
            wandb.init(reinit=True, project='weather-forecast')

    def run(self, run_mode):
        """Take run mode from choosen RUN argument."""
        if run_mode == 'train':
            self.train()
        elif run_mode == 'test':
            self.test()

    def train(self):
        """Train the data."""

        # Split the dataset into training set, validation set and test set
        train_df, val_df, test_df = train_test_split(self.dataset.df)

        # Standardize the data
        standard = Standardization(self.__C)
        train_df = standard.fit_transform(train_df)
        val_df = standard.transform(val_df)
        test_df = standard.transform(test_df)

        # Create window
        window = WindowGenerator(input_width=self.__C.N_HISTORY_DATA,
                                 label_width=self.__C.N_PREDICT_DATA,
                                 shift=self.__C.N_PREDICT_DATA,
                                 label_columns=self.__C.LABEL_COLUMNS,
                                 table_columns=self.dataset.df.columns,
                                 train_df=train_df, test_df=test_df,
                                 val_df=val_df)

        model = Factory(self.__C)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=self.__C.PATIENCE,
                                                          mode='min')

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                                              filepath=self.__C.CKPTS_FILE,
                                                              verbose=0,
                                                              save_weights_only=True,
                                                              save_best_only=True)

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam())
        model.fit(window.train, window.val, callbacks=[
                  early_stopping, model_checkpoint])

    def test(self):
        """Predict new data."""

        model_path = (self.__C.MODEL + '_' + str(self.__C.N_HISTORY_DATA) +
                      '_' + str(self.__C.N_PREDICT_DATA))
        if model_path not in os.listdir(self.__C.CKPTS_PATH):
            raise Exception("""
                You have not train the module with this configuration.
                Train a model or change your configuration.
                """)

        model = tf.keras.models.load_model(self.__C.CKPTS_PATH + model_path)

        if self.dataset.df.shape[0] < self.__C.N_HISTORY_DATA:
            raise Exception(
                f"""The provided dataset must have number of records greater or
                equal to {self.__C.N_HISTORY_DATA}""")

        # Standardize the data
        standard = Standardization(self.__C)
        X_new = standard.transform(self.dataset.df)
        # When predict data has more row than history data
        X_new = X_new[-self.__C.N_HISTORY_DATA:]

        # Change into tensor dataset and make prediction
        ds = tf.constant([np.array(X_new)])
        prediction = model(ds)

        # Inverse transform to get the real value
        X_pred = standard.inverse_transform(
            prediction.numpy()[0, :, :], self.__C.LABEL_COLUMNS)

        self.export(X_pred)

    def export(self, X_pred):
        """
        Export predict result to .csv file.
        """
        # Label the indices
        # Take the last index to begin labeling future index
        last = self.dataset.df.index[-1]
        indices = []

        for i in range(self.__C.N_PREDICT_DATA):
            indices.append(last + datetime.timedelta(minutes=10*i))

        df_pred = pd.DataFrame(X_pred, columns=self.__C.LABEL_COLUMNS)

        df_pred['Date Time'] = indices
        df_pred['Date Time'] = df_pred['Date Time'].dt.strftime(
            "%Y-%m-%d %H:%M:%S")
        df_pred.set_index('Date Time', inplace=True)
        name, _ = self.__C.TEST_FILENAME.split('.')

        if self.__C.EXPORT_MODE == 'csv':
            df_pred.to_csv(os.path.join(
                self.__C.PRED_PATH,
                f'{name}_{self.__C.MODEL}_{self.__C.N_HISTORY_DATA}_{self.__C.N_PREDICT_DATA}_pred.{self.__C.EXPORT_MODE}'),
                index_label='Date Time'
            )
        elif self.__C.EXPORT_MODE == 'json':
            df_pred.to_json(os.path.join(
                self.__C.PRED_PATH,
                f'{name}_{self.__C.MODEL}_{self.__C.N_HISTORY_DATA}_{self.__C.N_PREDICT_DATA}_pred.{self.__C.EXPORT_MODE}'),
                orient='index',
                indent=4
            )
