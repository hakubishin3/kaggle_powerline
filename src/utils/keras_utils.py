import os
import numpy as np
from glob import glob
from keras.utils import Sequence


def getNewestModel(model, dirname):
    """get the newest model file within a directory"""
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model


class DataGenerator(Sequence):
    def __init__(self, x, y, batch_size=64, cyclic_shift__alpha=0.5, skew__skew=0.05):
        self._x = x
        self._y = y
        self.indices = np.arange(y.shape[0])
        self.indices_mixup = np.arange(y.shape[0])
        self._batch_size = batch_size
        self._cyclic_shift__alpha = cyclic_shift__alpha
        self.skew__skew = skew__skew

    def __getitem__(self, index):
        indexes = self.indices[index * self._batch_size:(index + 1) * self._batch_size]
        indexes_mixup = self.indices_mixup[index * self._batch_size:(index + 1) * self._batch_size]
        batch_x, batch_y = [], []

        i_mixup = 0
        for _x, _y in zip(self._x[indexes], self._y[indexes]):
            _x = self.cyclic_shift(_x, self._cyclic_shift__alpha)
            _x = self.skew(_x, self.skew__skew)
            _x, _y = self.mixup(_x, _y, self._x[indexes_mixup[i_mixup]], self._y[indexes_mixup[i_mixup]])

            batch_x.append(_x)
            batch_y.append(_y)
            i_mixup += 1

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        np.random.shuffle(self.indices_mixup)

    def __len__(self):
        return len(self._y) // self._batch_size

    @staticmethod
    def cyclic_shift(x, alpha=0.5):
        s = np.random.uniform(0, alpha)
        part = int(len(x) * s)
        x_ = x[:part, :]
        _x = x[-len(x) + part:, :]
        return np.concatenate([_x, x_], axis=0)

    @staticmethod
    def skew(x, skew=0.05):
        s = 1 + np.random.normal(loc=0, scale=skew)
        return np.clip(x * s, -1, 1)

    @staticmethod
    def mixup(x, y, x_mix, y_mix, alpha=0.2):
        l = np.random.beta(alpha, alpha)
        x = x * l + x_mix * (1 - l)
        y = y * l + y_mix * (1 - l)
        return x, y
