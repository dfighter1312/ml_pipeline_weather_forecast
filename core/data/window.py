import tensorflow as tf
import numpy as np


class WindowGenerator():
    """
    Create a window of consecutive samples of data.
    Use to make a set of predictions which can benefit training.

    Attributes:
    -----------
        input_width (int):
            Number of consecutive history data.
        label_width (int):
            Number of future data we want to predict/show.
            If it is None, all the future data from 'shift' is shown.
        shift (int):
            Number of consecutive future data.
        train_df (pd.DataFrame):
            Training data.
        val_df (pd.DataFrame):
            Validation data.
        test_df (pd.DataFrame):
            Test data.
        label_columns (list of str, pd.Index or None):
            List of columns that we want to predict.
    """

    def __init__(self, input_width, shift,
                 train_df, test_df, val_df,
                 table_columns,
                 label_width=None,
                 label_columns=None):
        # Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices
        self.table_columns = table_columns
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(table_columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # slice(a, b) function will return a slice object.
        # arr[slice(a, b)] = arr[a:b]
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """Convert list of consecutive inputs into a window of inputs and a window of labels (output)."""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.`
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """Convert the DataFrame into a tf.data.Dataset."""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
