import os

def get_model_path(__C, load=True):
    path = '_'.join([__C.DATA_CLASS, __C.MODEL, str(__C.N_HISTORY_DATA), str(__C.N_PREDICT_DATA)])
    folder = __C.CKPTS_PATH if not load else __C.CKPTS_OUTPUT_PATH # Fix for pachyderm update
    if load:
        check_path(folder, path)
    return folder + path

def check_path(folder, file):
    if file not in os.listdir(folder):
            raise Exception("""
                You have not train the module with this configuration.
                Train a model or change your configuration.
                """)