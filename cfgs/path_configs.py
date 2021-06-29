import os

class PATH:

    def __init__(self):

        self.DATASET_PATH = './datasets/'
        self.PRED_PATH = './results/pred/'
        self.CKPTS_PATH = './ckpts/'

        self.init_path()

    def init_path(self):
        """Initialize the directory in case there are none.""" 
        
        if 'pred' not in os.listdir('./results'):
            os.mkdir(f'./results/pred')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')

        self.DATA_PATH = {
            'jena': {
                'train': self.DATASET_PATH + 'jena/mpi_roof_2020a.csv',
                'test': self.DATASET_PATH + 'jena/mpi_roof_2020b.csv'
            },
            'bewaco': {
                'train': self.DATASET_PATH + 'real/export-reec56.BEWACO 2021.csv',
                'test': self.DATASET_PATH + 'real/export-reec56.BEWACO 2021.csv'
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