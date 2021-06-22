from core.model.base_model import BaseModel
import tensorflow as tf
import wandb

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
                        optimizer=tf.optimizers.Adam(learning_rate=self.__C.LEARNING_RATE))

        self.linear.fit(X_train, epochs=self.__C.MAX_EPOCHS, validation_data=X_val,
                        callbacks=self.__C.callbacks)
        if self.__C.wandb:
            wandb.log({
                'train_loss': self.linear.evaluate(X_train),
                'val_loss': self.linear.evaluate(X_val)
            })
        self.linear.save(self.__C.CKPTS_PATH + self.__C.MODEL)