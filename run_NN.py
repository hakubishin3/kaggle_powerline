import json
import gc
import os
import random as rn
import tensorflow as tf
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from src.utils.logger_functions import get_module_logger
from src.utils.json_dump import save_json
from src.data.load_dataset import load_metadata, load_tsdata
from src.data.wavelet_denoising import add_high_pass_filter
from src.features.base import load_features
from src.models.cnn import get_model
from keras.optimizers import Nadam, SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import keras.backend as K
from src.models.get_folds import get_StratifiedKFold, get_adversarial_validation
from keras.models import load_model
from src.utils.metric import eval_mcc, CalcBestMcc, CalcAUC, matthews_correlation, CalcBestMcc
from src.utils.keras_utils import getNewestModel, DataGenerator
from sklearn.metrics import log_loss, roc_auc_score
from memory_profiler import profile


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./configs/cnn_0.json')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--out', '-o', default='model_v0.9')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))
    args_log = {"args": {
        "config": args.config,
        "debug": args.debug,
        "out": args.out
    }}
    config.update(args_log)

    # load meta-data
    train_path = config["dataset"]["input_directory"] + config["dataset"]["files"]["meta"]["train"]
    test_path = config["dataset"]["input_directory"] + config["dataset"]["files"]["meta"]["test"]
    train_meta, test_meta = load_metadata(train_path, test_path, args.debug)
    logger.debug(f'train_meta: {train_meta.shape}')

    # load input data
    x_train = np.load("./data/interim/x_train_1dcnn_v1.5.npy")
    y_train = np.load("./data/interim/y_train_1dcnn_v1.5.npy")
    logger.debug(f'x_train: {x_train.shape}, y_train: {y_train.shape}')

    # get param value
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    lr = config["model"]["lr"]
    dropout = config["model"]["dropout"]
    k = config["model"]["k"]
    cyclic_shift__alpha = config["model"]["cyclic_shift__alpha"]
    skew__skew = config["model"]["skew__skew"]
    repetitions = config["model"]["repetitions"]
    early_stopping_patience = config["model"]["early_stopping_patience"]

    # make save dir
    out_path = f"./data/output/cnn/{args.out}/"
    save_path = f"./data/output/cnn/{args.out}/save/"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        os.mkdir(save_path)

    # get folds and train
    if config["cv"]["method"] == "StratifiedKFold":
        n_splits = config["cv"]["n_splits"]
        random_state = config["cv"]["random_state"]
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(x_train, y_train))
    elif config["cv"]["method"] == "AdversarialValidation":
        folds = get_adversarial_validation(train_meta)

    logger.info("train model")
    oof_proba = np.zeros(len(x_train))
    models = []
    history_list = []
    val_best_mcc_list = []
    val_best_thr_list = []
    best_model_name_list = []
    val_best_auc_list = []
    val_best_loss_list = []
    val_best_mcc50_list = []

    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        logger.info(f"{i_fold + 1}fold.")

        """
        trn_gen = DataGenerator(
            x_train[trn_idx], y_train[trn_idx], batch_size=batch_size,
            cyclic_shift__alpha=cyclic_shift__alpha, skew__skew=skew__skew
        )
        """
        logger.debug(f'positive ratio of train: {y_train[trn_idx].sum() / len(y_train[trn_idx])}, {y_train[trn_idx].sum()} / {len(y_train[trn_idx])}')
        logger.debug(f'positive ratio of valid: {y_train[val_idx].sum() / len(y_train[val_idx])}, {y_train[val_idx].sum()} / {len(y_train[val_idx])}')

        optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.0)
        model = get_model(input_size=(x_train.shape[1], x_train.shape[2]), dropout=dropout, k=k, repetitions=repetitions)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[matthews_correlation])
        print(model.summary())

        # setting callbacks
        chkpt = os.path.join(save_path, 'weights_fold%s_epoch{epoch:02d}_valmcc50t{val_matthews_correlation:.3f}_valloss{val_loss:.3f}.hdf5' % (i_fold + 1))
        checkpointer = ModelCheckpoint(filepath=chkpt, verbose=1, save_best_only=False, monitor='val_matthews_correlation', mode="max")
        early_stopping = EarlyStopping(monitor='val_matthews_correlation', patience=early_stopping_patience, mode="max")
        best_mcc = CalcBestMcc()

        # train
        history = model.fit_generator(
            trn_gen, steps_per_epoch=len(y_train[trn_idx]) // batch_size,
            epochs=epochs, callbacks=[best_mcc, callbacks_auc, early_stopping, checkpointer],
            validation_data=(x_train[val_idx], y_train[val_idx]),
            verbose=1, shuffle=True, max_queue_size=1
        )

        history_list.append(history.history)

        # get checkpoint model
        best_model = getNewestModel(model, save_path)
        models.append(best_model)

        # check score
        y_proba = best_model.predict(x_train[val_idx]).reshape(-1)
        oof_proba[val_idx] = y_proba
        best_proba, best_mcc, y_pred = eval_mcc(y_train[val_idx], y_proba, show=True)
        auc = roc_auc_score(y_train[val_idx], y_proba)
        logger.debug(f'Best threshold: {best_proba}, Best mcc: {best_mcc}, Best auc: {auc}')
        val_best_mcc_list.append(best_mcc)
        val_best_thr_list.append(best_proba)
        val_best_auc_list.append(auc)

        score = best_model.evaluate(x_train[val_idx], y_train[val_idx])
        best_model_save_name = f"best_weights_fold{i_fold+1}_valmcc{best_mcc:.3f}_valmcc50t{score[1]:.3f}_valauc{auc:.3f}_valloss{score[0]:.3f}.hdf5"
        best_model.save(save_path + best_model_save_name)
        best_model_name_list.append(best_model_save_name)
        val_best_loss_list = [score[0]]
        val_best_mcc50_list = [score[1]]

    # check score of oof
    best_proba, best_mcc, oof_preds = eval_mcc(y_train, oof_proba, show=True)
    logger.debug(f'Best threshold: {best_proba}, Best metric: {best_mcc}')

    # add log
    train_results = {"evals_result": {
        "oof_best_mcc": best_mcc,
        "oof_best_thr": best_proba,
        "cv_best_mcc": {f"cv{i+1}": best_mcc for i, best_mcc in enumerate(val_best_mcc_list)},
        "cv_best_thr": {f"cv{i+1}": best_thr for i, best_thr in enumerate(val_best_thr_list)},
        "cv_best_loss": {f"cv{i+1}": loss for i, loss in enumerate(val_best_loss_list)},
        "cv_best_mcc50": {f"cv{i+1}": mcc50 for i, mcc50 in enumerate(val_best_mcc50_list)},
        "cv_best_auc": {f"cv{i+1}": auc for i, auc in enumerate(val_best_auc_list)},
        "best_model_name": {f"cv{i+1}": best_model_name for i, best_model_name in enumerate(best_model_name_list)},
        "history": {f"history{i+1}": history for i, history in enumerate(history_list)}
    }}
    config.update(train_results)

    # save
    train_meta["proba"] = oof_proba
    train_meta["preds"] = oof_preds
    train_meta.to_csv(out_path + f'oof_proba_{args.out}.csv', header=True, index=False)
    save_json(config, args.out, logger, save_path=out_path + f'output_{args.out}.json')


if __name__ == '__main__':
    main()
