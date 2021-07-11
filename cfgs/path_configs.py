import os

class PATH:

    def __init__(self):

        self.TRAIN_PATH = '/pfs/input_train/'
        self.TEST_PATH = '/pfs/input_test/'
        self.PRED_PATH = '/pfs/out/result/'
        self.CKPTS_PATH = '/pfs/out/ckpts/'
        self.CKPTS_OUTPUT_PATH = '/pfs/model/ckpts/'

        self.init_path()

    def init_path(self):
        """Initialize the directory in case there are none.""" 
        
        if 'result' not in os.listdir('/pfs/out'):
            os.mkdir('/pfs/out/result')

        if 'ckpts' not in os.listdir('/pfs/out'):
            os.mkdir('/pfs/out/ckpts')

        self.DATA_PATH = {
            'jena': {
                'train': self.TRAIN_PATH + 'mpi_roof_2020a.csv',
                'test': self.TEST_PATH + 'mpi_roof_2020b.csv'
            },
            'bewaco': {
                'train': self.TRAIN_PATH + 'export-reec56.BEWACO 2021.csv',
                'test': self.TEST_PATH + 'export-reec56.BEWACO 2021.csv'
            }
        }

        self.CKPTS_FILE = self.CKPTS_PATH + 'model_checkpoint.h5'

    def check_path(self):
        """Check whether the dataset is appear."""

        for mode in self.DATA_PATH:
            if not os.path.exists(self.TARGET_PATH[mode]):
                print(self.DATA_PATH[mode] + ' IS NOT EXIST')
                exit(-1)
        
        print('Check path finished')