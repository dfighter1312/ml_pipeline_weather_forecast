from core.model.base_model import BaseModel
import tensorflow as tf
import wandb


class MLP(BaseModel):

    def __init__(self, __C):
        self.__C = __C

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=self.__C.LAYER_1_UNITS, activation='relu'),
            tf.keras.layers.Dense(
                units=self.__C.LAYER_2_UNITS, activation='relu'),
            tf.keras.layers.Dense(
                units=self.__C.N_PREDICT_DATA * self.__C.N_FEATURES),
            tf.keras.layers.Reshape(
                [self.__C.N_PREDICT_DATA, self.__C.N_FEATURES]),
        ])

    def compile_and_fit(self, X_train, X_val):
        self.mlp.compile(loss=tf.losses.MeanSquaredError(),
                         optimizer=tf.optimizers.Adam())

        self.mlp.fit(X_train, epochs=self.__C.MAX_EPOCHS, validation_data=X_val, verbose=0,
                     callbacks=self.__C.callbacks)

        if self.__C.wandb:
            wandb.log({
                'train_loss': self.mlp.evaluate(X_train),
                'val_loss': self.mlp.evaluate(X_val)
            })

        self.mlp.save(self.__C.CKPTS_PATH + self.__C.MODEL)
