from re import X
from core.model.base_model import BaseModel
import tensorflow as tf

class LSTM(BaseModel):

    def __init__(self, __C):
        self.__C = __C
        self.lstm = tf.keras.Sequential([
                                tf.keras.layers.LSTM(__C.LSTM_UNITS, return_sequences=False),
                                tf.keras.layers.Dense(self.__C.N_PREDICT_DATA * self.__C.N_FEATURES),
                                tf.keras.layers.Reshape([self.__C.N_PREDICT_DATA, self.__C.N_FEATURES])
                            ])

    def compile_and_fit(self, X_train, X_val):
        self.lstm.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam())

        self.lstm.fit(X_train, epochs=self.__C.MAX_EPOCHS, validation_data=X_val, verbose=0,
                        callbacks=[self.__C.early_stopping, self.__C.model_checkpoint])

        self.lstm.save(self.__C.CKPTS_PATH + self.__C.MODEL)