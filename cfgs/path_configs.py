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

        self.TRAIN_FILENAME = 'mpi_roof_2020a.csv'
        self.TEST_FILENAME = 'mpi_roof_2020b.csv'

        self.DATA_PATH = {
            'train': self.DATASET_PATH + self.TRAIN_FILENAME,
            'test': self.DATASET_PATH + self.TEST_FILENAME
        }

    def check_path(self):
        """Check whether the dataset is appear."""

        for mode in self.DATA_PATH:
            if not os.path.exists(self.TARGET_PATH[mode]):
                print(self.DATA_PATH[mode] + ' IS NOT EXIST')
                exit(-1)
        
        print('Check path finished')