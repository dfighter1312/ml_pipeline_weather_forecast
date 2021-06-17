import numpy as np
import pandas as pd
import os
import datetime
import tensorflow as tf
from core.utils.train_test_split import train_test_split
from core.data.dataset import Dataset
from core.data.window import WindowGenerator
from core.utils.standardize import Standardization
from core.model.linear import Linear
from core.model.lstm import LSTM
from core.model.mlp import MLP

class Execution():

    def __init__(self, __C):
        self.__C = __C
        self.dataset = Dataset(self.__C)
    
    def train(self):
        
        self.dataset.preprocess()

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
                                      train_df=train_df, test_df=test_df, val_df=val_df)

        
        path_checkpoint = self.__C.CKPTS_PATH + 'model_checkpoint.h5'
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=self.__C.PATIENCE,
                                                          mode='min')

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor="val_loss", 
                                                              filepath=path_checkpoint,
                                                              verbose=0,
                                                              save_weights_only=True,
                                                              save_best_only=True)

        setattr(self.__C, 'early_stopping', early_stopping)
        setattr(self.__C, 'model_checkpoint', model_checkpoint)

        model = self.select_model()

        model.compile_and_fit(window.train, window.val)

    def test(self):

        self.dataset.preprocess()

        model = tf.keras.models.load_model(self.__C.CKPTS_PATH + self.__C.MODEL)

        if self.dataset.df.shape[0] < self.__C.N_HISTORY_DATA:
            raise Exception(f'The provided dataset must have number of records greater or equal to {self.__C.N_HISTORY_DATA}')
        
        # Standardize the data
        standard = Standardization(self.__C)
        X_new = standard.transform(self.dataset.df)
        X_new = X_new[-self.__C.N_HISTORY_DATA:]
        
        # Change into tensor dataset and make prediction
        ds = tf.constant([np.array(X_new)])
        prediction = model(ds)
        
        # Inverse transform to get the real value
        X_pred = standard.inverse_transform(prediction.numpy()[0, :, :], self.__C.LABEL_COLUMNS)

        last = self.dataset.df.index[-1]
        indices = []
        
        for i in range(self.__C.N_PREDICT_DATA):
            indices.append(last + datetime.timedelta(minutes=10*i))
        
        df_pred = pd.DataFrame(X_pred, columns=self.__C.LABEL_COLUMNS, index=indices)

        name, tag = self.__C.TEST_FILENAME.split('.')

        df_pred.to_csv(os.path.join(self.__C.PRED_PATH, f'{name}_{self.__C.MODEL}_pred.{tag}'), index_label='Date Time')


    def select_model(self):
        if self.__C.MODEL == 'linear':
            return Linear(self.__C)
        elif self.__C.MODEL == 'lstm':
            return LSTM(self.__C)
        elif self.__C.MODEL == 'mlp':
            return MLP(self.__C)
        else:
            raise ValueError(f'No model name {self.__C.MODEL} is implemented')

    def run(self, run_mode):
        if run_mode == 'train':
            self.train()
        elif run_mode == 'test':
            self.test()