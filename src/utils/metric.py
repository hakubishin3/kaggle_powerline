import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, matthews_corrcoef


class CalcBestMcc(Callback):
    """calculate mcc using by validation data"""
    def on_epoch_end(self, epoch, logs):
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        y_pred = np.asarray(self.model.predict(x_val))
        y_val = y_val.reshape(-1)
        y_pred = y_pred.reshape(-1)

        score = matthews_corrcoef(y_val, (y_pred > 0.5).astype(int))
        logs['val_matthews_correlation_end'] = score
        print(f"val_matthews_correlation_end: {score}")
        return


class CalcAUC(Callback):
    """calculate AUC using by validation data"""
    def on_epoch_end(self, epoch, logs):
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        y_pred = np.asarray(self.model.predict(x_val))
        y_val = y_val.reshape(-1)
        y_pred = y_pred.reshape(-1)

        auc = roc_auc_score(y_val, y_pred)
        logs['val_auc'] = auc
        print(f"val_auc: {auc}")
        return


def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
