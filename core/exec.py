import numpy as np
import tensorflow as tf
import wandb
from core.data.jena import JenaDataset
from core.data.bewaco import BewacoDataset
from core.data.window import WindowGenerator
from core.model.factory import ModelFactory
from core.utils.standardize import Standardization
from core.utils.train_test_split import train_test_split
from core.utils.os_settings import get_model_path
from core.utils.export import export


class Execution():
    """Execution class. Part of the entrypoint."""

    def __init__(self, __C):
        self.__C = __C
        self.dataset = None
        if __C.DATA_CLASS == 'jena':
            self.dataset = JenaDataset(self.__C)
        elif __C.DATA_CLASS == 'bewaco':
            self.dataset = BewacoDataset(self.__C)
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
        print(train_df, val_df, test_df)

        # Create window
        window = WindowGenerator(input_width=self.__C.N_HISTORY_DATA,
                                        label_width=self.__C.N_PREDICT_DATA,
                                        shift=self.__C.N_PREDICT_DATA,
                                        label_columns=self.dataset.get_str_label_columns(),
                                        table_columns=self.dataset.get_columns(),
                                        train_df=train_df, test_df=test_df,
                                        val_df=val_df)

        model = ModelFactory(self.__C)

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

        model_path = (get_model_path(self.__C))
        model = tf.keras.models.load_model(model_path)

        self.dataset.set_predict_dataset()

        # Standardize the data
        standard = Standardization(self.__C)
        X = standard.transform(self.dataset.df)

        # Change into tensor dataset and make prediction
        ds = tf.constant([np.array(X)])
        prediction = model(ds)

        # Inverse transform to get the real value
        X_pred = standard.inverse_transform(
            prediction.numpy()[0, :, :], self.__C.LABEL_COLUMNS)

        export(self.__C, self.dataset, X_pred)