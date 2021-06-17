import numpy as np
import argparse
import os
from cfgs.base_configs import Configs
from core.exec import Execution

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Weather Prediction arguments')
    parser.add_argument('--RUN', dest='RUN_MODE',
                        choices=['train', 'test'],
                        type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    __C = Configs()
    
    args = parse_args()
    args_dict = __C.parse_to_dict(args)
    __C.add_args(args_dict)
    __C.proc()
    
    print('Hyperparameters:')
    print(__C)

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)