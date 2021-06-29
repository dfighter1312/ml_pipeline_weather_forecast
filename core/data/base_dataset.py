from abc import abstractmethod, ABC


class BaseDataset(ABC):

    @abstractmethod
    def __init__(self, __C):
        """
        Initialize the DataFrame from the file link.

        Args:
        ------
            __C: Config
                Configuration file.
        """
        raise NotImplementedError

    @abstractmethod
    def create_dataframe(self):
        """Get the DataFrame from CSV file."""
        raise NotImplementedError

    @abstractmethod
    def get_columns(self):
        """Get columns of DataFrame"""
        raise NotImplementedError

    @abstractmethod
    def get_str_label_columns(self):
        """Convert all integer indices to named indices."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self):
        """
        Perform preprocessing data for data preparation.
        (Data cleaning and feature engineering).
        """
        raise NotImplementedError