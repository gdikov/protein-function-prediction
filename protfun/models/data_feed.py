class DataFeeder(object):
    def __init__(self):
        self.samples_per_class = 1000

    def iterate_test_data(self):
        return None

    def iterate_train_data(self):
        return None

    def iterate_val_data(self):
        return None

    def set_samples_per_class(self, samples_per_class):
        self.samples_per_class = samples_per_class


class MemmapFeeder(DataFeeder):
    def iterate_test_data(self):
        pass

    def iterate_train_data(self):
        pass

    def iterate_val_data(self):
        pass


class GridFeeder(DataFeeder):
    def iterate_test_data(self):
        pass

    def iterate_train_data(self):
        pass

    def iterate_val_data(self):
        pass
