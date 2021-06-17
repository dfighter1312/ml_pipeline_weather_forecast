from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(self):
        """
        Initialization
        
        Args:
        -------
            __C: Configs
                Configuration class.
        """
        raise NotImplementedError

    @abstractmethod
    def compile_and_fit(self):
        """
        Compile and fit the model, then save the model into checkpoints.
        """
        raise NotImplementedError
