def train_test_split(X, train_size=0.7, val_size=0.2):
        n = len(X)
        
        if train_size + val_size > 1.0:
            raise ValueError('train_size + val_size must be smaller than 1.0')
            
        train_df = X[0 : int(n * train_size)]
        val_df = X[int(n * train_size) : int(n * (train_size + val_size))]
        test_df = X[int(n * (train_size + val_size)):]

        return train_df, val_df, test_df