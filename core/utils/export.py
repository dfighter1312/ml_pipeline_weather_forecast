import os
import datetime
import pandas as pd


def export(__C, dataset, X_pred):
    """
    Export predict result to .csv file.
    """
    # Label the indices
    # Take the last index to begin labeling future index
    last_index = dataset.df.index[-1]
    indices = []

    for i in range(__C.N_PREDICT_DATA):
        indices.append(last_index + datetime.timedelta(minutes=10*i))

    df_pred = pd.DataFrame(X_pred, columns=dataset.get_str_label_columns())

    df_pred['Date Time'] = indices
    df_pred['Date Time'] = df_pred['Date Time'].dt.strftime(
        "%Y-%m-%d %H:%M:%S")
    df_pred.set_index('Date Time', inplace=True)

    if __C.EXPORT_MODE == 'csv':
        df_pred.to_csv(os.path.join(
            __C.PRED_PATH,
            f'{__C.DATA_CLASS}_{__C.MODEL}_{__C.N_HISTORY_DATA}_{__C.N_PREDICT_DATA}_pred.{__C.EXPORT_MODE}'),
            index_label='Date Time'
        )
    elif __C.EXPORT_MODE == 'json':
        df_pred.to_json(os.path.join(
            __C.PRED_PATH,
            f'{name}_{__C.MODEL}_{__C.N_HISTORY_DATA}_{__C.N_PREDICT_DATA}_pred.{__C.EXPORT_MODE}'),
            orient='index',
            indent=4
        )