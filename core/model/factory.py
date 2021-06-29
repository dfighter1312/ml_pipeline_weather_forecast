from core.model.linear import Linear
from core.model.lstm import LSTM
from core.model.mlp import MLP


def ModelFactory(__C):
    """
    Factory Method.
    Select the model base on the choice in configuration file.
    """

    models = {
        "linear": Linear,
        "lstm": LSTM,
        "mlp": MLP,
    }

    return models[__C.MODEL](__C)