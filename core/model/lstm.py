from core.model.base_model import BaseModel
from core.utils.os_settings import get_model_path
import tensorflow as tf
import wandb


class LSTM(BaseModel):

    def __init__(self, __C):
        self.__C = __C
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.LSTM(__C.LSTM_UNITS, return_sequences=False),
        #     tf.keras.layers.Dense(
        #         self.__C.N_PREDICT_DATA * self.__C.N_FEATURES),
        #     tf.keras.layers.Reshape(
        #         [self.__C.N_PREDICT_DATA, self.__C.N_FEATURES])
        # ])
        self.model = tf.keras.Sequential()
        for i, unit in enumerate(__C.LSTM_UNITS):
            self.model.add(tf.keras.layers.LSTM(unit, return_sequences=(i != len(__C.LSTM_UNITS) - 1)))
        self.model.add(tf.keras.layers.Dense(self.__C.N_PREDICT_DATA * self.__C.N_FEATURES))
        self.model.add(tf.keras.layers.Reshape([self.__C.N_PREDICT_DATA, self.__C.N_FEATURES]))

    def compile(self, loss, optimizer):
        self.model.compile(loss=loss,
                            optimizer=optimizer)

    def fit(self, X_train, X_val, callbacks):
        self.model.fit(X_train, epochs=self.__C.MAX_EPOCHS, validation_data=X_val,
                        callbacks=callbacks)

        if self.__C.wandb:
            train_loss = self.model.evaluate(X_train)
            val_loss = self.model.evaluate(X_val)
            self._wandb_save(train_loss, val_loss)
        
        self._save_model()

    def _wandb_save(self, train_loss, val_loss):
        wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss
        })

    def _save_model(self):        
        self.model.save(get_model_path(self.__C, load=False))
