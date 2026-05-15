class SafeTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, gap: int = 1, min_train_size: int = 500):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1.")
        if gap < 0:
            raise ValueError("gap must be non-negative.")
        if min_train_size < 1:
            raise ValueError("min_train_size must be at least 1.")
        self.n_splits = n_splits
        self.gap = gap
        self.min_train_size = min_train_size
        self.shuffle = False

    def split(self, X):
        n_samples = len(X)
        test_size = max(1, (n_samples - self.min_train_size - self.gap) // self.n_splits)
        if n_samples < self.min_train_size + self.gap + test_size:
            raise ValueError("Not enough samples for safe time-series split.")
        for fold in range(self.n_splits):
            train_end = self.min_train_size + fold * test_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            if test_start >= n_samples or test_end <= test_start:
                break
            yield list(range(0, train_end)), list(range(test_start, test_end))
