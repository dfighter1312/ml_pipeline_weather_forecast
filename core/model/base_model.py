from abc import abstractmethod, ABC


class BaseModel(ABC):

    @abstractmethod
    def __init__(self, __C):
        """
        Initialization

        Args:
        -------
            __C: Configs
                Configuration class.
        """
        raise NotImplementedError

    @abstractmethod
    def compile(self, loss, optimizer):
        """
        Compile the model by defining loss function and optimizer.
        
        Args:
        -------
            loss: tf.keras.losses
                Model's loss function.
            optimizer: tf.keras.optimizer
                Model's optimizer.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_train, X_val, callbacks=None):
        """
        Fit the model into X_train (using X_val to control number of epochs).

        Args:
        -------
            X_train: np.array
                Training data.
            X_val: np.array
                Validation data.
            callback: 
                List of callbacks.
        """
        raise NotImplementedError

    @abstractmethod
    def _wandb_save(self, train_loss, val_loss):
        """
        Save the model train_loss and val_loss into Weight & Biases server.
        Use for hyperparameter tuning.
        To activate, you need set --wandb argument to True.

        Args:
        -------
            train_loss: double
                Training loss recorded at best model (not the last one).
            val_loss: double
                Validation loss recorded at best model.
        """
        raise NotImplementedError

    @abstractmethod
    def _save_model(self):        
        """
        Save the model into the local repository.
        The file path is saved in Config object.
        """
        raise NotImplementedError
