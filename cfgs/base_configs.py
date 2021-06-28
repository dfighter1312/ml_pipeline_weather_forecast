from types import MethodType
from cfgs.path_configs import PATH

class Configs(PATH):

    def __init__(self):
        super(Configs, self).__init__()
        # self.SEED = random.randint(0, 999999)
        # self.VERSION = str(self.SEED)
        self.init_params()

    def init_params(self):
        """Initialize parameters for all models."""
        
        self.MODEL = 'linear'

        self.N_HISTORY_DATA = 18

        self.N_PREDICT_DATA = 6

        self.LABEL_COLUMNS = ['T (degC)', 'p (mbar)', 'sh (g/kg)']

        self.MAX_EPOCHS = 10

        self.PATIENCE = 3

        self.EXPORT_MODE = 'csv'

        self.wandb = False

        # Linear config
        self.L1_REGULARIZE = 0.01

        # MLP config
        self.LAYER_1_UNITS = 8
        
        self.LAYER_2_UNITS = 8

        # LSTM config
        self.LSTM_UNITS = 8

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('__') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def proc(self):
        self.N_FEATURES = len(self.LABEL_COLUMNS)
        assert self.RUN_MODE in ['train', 'test']


    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''