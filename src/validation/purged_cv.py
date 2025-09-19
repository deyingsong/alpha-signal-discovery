class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, test_size=90, gap=5, purge_length=10):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.purge_length = purge_length
    
    def split(self, X, y=None):
        n_samples = len(X)
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * self.test_size
            test_end = test_start + self.test_size
            
            # Purge training data that's too close to test set
            train_end = test_start - self.gap - self.purge_length
            
            if train_end > 100:  # Ensure minimum training size
                train_indices = np.arange(0, train_end)
                test_indices = np.arange(test_start, min(test_end, n_samples))
                yield train_indices, test_indices