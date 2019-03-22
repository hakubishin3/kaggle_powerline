import pandas as pd
import multiprocessing
import pyarrow.parquet as pq
from pathlib import Path


def load_metadata(
        train_path="../../data/input/metadata_train.csv",
        test_path="../../data/input/metadata_test.csv",
        debug=False):

    train = pd.read_csv(train_path)

    if debug is False:
        test = pd.read_csv(test_path)
    else:
        test = None

    return train, test


def load_tsdata(
        train_path="../../data/input/train.parquet",
        test_path="../../data/input/test.parquet",
        debug=False):

    n_cpu = multiprocessing.cpu_count()
    train = pq.ParquetDataset(train_path).read(nthreads=n_cpu).to_pandas().transpose()
    train.index = range(0, len(train))

    if debug is False:
        test = pq.ParquetDataset(test_path).read(nthreads=n_cpu).to_pandas().transpose()
        test.index = range(0, len(test))
    else:
        test = None

    return train, test
