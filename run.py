import argparse
from cfgs.base_configs import Configs
from core.exec import Execution


def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(
        description='Weather Prediction arguments')
    parser.add_argument('--RUN', dest='RUN_MODE',
                        choices=['train', 'test'],
                        type=str, required=True)

    parser.add_argument("--MODEL", type=str, default="linear")
    parser.add_argument("--LABEL_COLUMNS", type=list,
                        default=['T (degC)', 'p (mbar)', 'sh (g/kg)'])
    parser.add_argument("--MAX_EPOCHS", type=int, default=10)
    parser.add_argument("--L1_REGULARIZE", type=float, default=0.01)
    parser.add_argument("--LAYER_1_UNITS", type=int, default=8)
    parser.add_argument("--LAYER_2_UNITS", type=int, default=8)
    parser.add_argument("--LSTM_UNITS", type=int, default=8)
    parser.add_argument("--PATIENCE", type=int, default=2)
    parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
    parser.add_argument("--N_HISTORY_DATA", type=int, default=18)
    parser.add_argument("--N_PREDICT_DATA", type=int, default=6)
    parser.add_argument("--EXPORT_MODE", type=str, default='csv')
    parser.add_argument("--wandb", type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    __C = Configs()
    print(parse_args())
    args_dict = __C.parse_to_dict(parse_args())
    __C.add_args(args_dict)
    __C.proc()

    print('Hyperparameters:')
    print(__C)

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)
