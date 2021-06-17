from core.model.base_model import BaseModel
import tensorflow as tf

class Linear(BaseModel):

    def __init__(self, __C):
        self.__C = __C
        self.linear = tf.keras.Sequential([
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(units=self.__C.N_PREDICT_DATA * self.__C.N_FEATURES, 
                                                      kernel_regularizer=tf.keras.regularizers.L1(__C.L1_REGULARIZE)),
                                tf.keras.layers.Reshape([self.__C.N_PREDICT_DATA, self.__C.N_FEATURES]),
                            ])

    def compile_and_fit(self, X_train, X_val):
        self.linear.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam())

        self.linear.fit(X_train, epochs=self.__C.MAX_EPOCHS, validation_data=X_val, verbose=0,
                        callbacks=[self.__C.early_stopping, self.__C.model_checkpoint])

        self.linear.save(self.__C.CKPTS_PATH + self.__C.MODEL)