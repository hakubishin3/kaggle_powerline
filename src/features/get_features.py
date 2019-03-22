import tsfresh
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences


def get_features(dataset='train', split_parts=10):
    if dataset == 'train':
        cache_file = 'X.npy'
        meta_file = '../data/input/metadata_train.csv'
    elif dataset == 'test':
        cache_file = 'X_test.npy'
        meta_file = '../data/input/metadata_test.csv'

    meta_df = pd.read_csv(meta_file)

    data_measurements = meta_df.pivot(
        index='id_measurement', columns='phase', values='signal_id'
    )
    data_measurements = data_measurements.values
    data_measurements = np.array_split(data_measurements, split_parts, axis=0)
    X = Parallel(n_jobs=min(split_parts, MAX_THREADS), verbose=1)(
        delayed(prep_data)(p, dataset) for p in data_measurements
    )
    X = np.concatenate(X, axis=0)

    if dataset == 'train':
        y = meta_df.loc[meta_df['phase'] == 0, 'target'].values
        np.save(save_dir + "X.npy", X)
        np.save(save_dir + "y.npy", y)
    elif dataset == 'test':
        y = None
        np.save(save_dir + "X_test.npy", X)

    return X, y


def prep_data(signal_ids, dataset="train"):
    signal_ids_all = np.concatenate(signal_ids)
    if dataset == "train":
        praq_data = pq.read_pandas(
            '../data/input/train.parquet', columns=[str(i) for i in signal_ids_all]).to_pandas()
    elif dataset == "test":
        praq_data = pq.read_pandas(
            '../data/input/test.parquet', columns=[str(i) for i in signal_ids_all]).to_pandas()
    else:
        raise ValueError("Unknown dataset")

    X = []
    for sids in signal_ids:
        data = praq_data[[str(s) for s in sids]].values.T
        X_signal = [transform_ts(signal) for signal in data]
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)

    return X


def transform_ts(ts, n_dim=160):
    # setting
    sample_size = 800000
    bucket_size = int(sample_size / n_dim)
    new_ts = []

    # signal processing
    ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    ts_std = high_pass_filter(ts_std, low_cutoff=10000)
    ts_std = denoise_signal(ts_std)

    for i in range(0, sample_size, bucket_size):
        ts_range = ts_std[i:i + bucket_size]

        # basic features
        std = ts_range.std()
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]

        # peak features
        pos_peaks, _ = find_peaks(ts_range, distance=50)
        neg_peaks, _ = find_peaks(ts_range * -1, distance=50)
        widths = np.diff(pos_peaks)
        prominences = peak_prominences(ts_range, pos_peaks)[0]

        peak_result = {
            'pos_peak_count': pos_peaks.size / 160,
            'peak_count_buchet': (pos_peaks.size + neg_peaks.size) / 160,
            'width_mean_buchet': (widths.mean() / bucket_size if widths.size else 1.) - 1,
            'width_max_buchet': (widths.max() / bucket_size if widths.size else 1.) - 1,
            'width_min_buchet': (widths.min() / bucket_size if widths.size else 1.) - 1,
            'prominence_mean_buchet': prominences.mean() / 2 if prominences.size else 0.,
            'prominence_max_buchet': prominences.max() / 2 if prominences.size else 0.,
            'prominence_min_buchet': prominences.min() / 2 if prominences.size else 0.,
        }
        peak_result = np.array(list(peak_result.values())).astype(float)

        # complexity
        fe_cid_ce = tsfresh.feature_extraction.feature_calculators.cid_ce(ts_range, normalize=False)
        fe_cid_ce = np.clip(fe_cid_ce, 0, 1)

        # sumamry
        new_ts.append(np.concatenate([
            np.asarray([fe_cid_ce]),
            peak_result,
            np.asarray([std, max_range]),
            percentil_calc
        ]))

    return np.asarray(new_ts)


if __name__ == '__main__':
    x_train, y_train = get_features(dataset='train', split_parts=10)
    x_test, _ = get_features(dataset='test', split_parts=10)
