import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import matthews_corrcoef


class LightGBM(object):
    def fit(self, d_train: lgb.Dataset, d_valid: lgb.Dataset, params: dict):
        evals_result = {}
        model = lgb.train(
            params=params['model_params'],
            train_set=d_train,
            valid_sets=[d_train, d_valid],
            valid_names=['train', 'valid'],
            evals_result=evals_result,
            **params['train_params']
        )
        return model, evals_result

    def cv(self, x_train, y_train, x_test, folds, params: dict):
        # init predictions
        oof_preds = np.zeros(len(x_train))
        sub_preds = np.zeros(len(x_test))
        importances = pd.DataFrame(index=x_train.columns)
        best_iteration = 0
        cv_score_list = []
        models = []

        # Run cross-validation
        n_folds = len(folds)

        for i_fold, (trn_idx, val_idx) in enumerate(folds):
            d_train = lgb.Dataset(x_train.iloc[trn_idx], label=y_train[trn_idx])
            d_valid = lgb.Dataset(x_train.iloc[val_idx], label=y_train[val_idx])
            print(f"positive ratio: {(y_train[trn_idx] == 1).sum() / len(y_train[trn_idx])}")

            # train model
            model, evals_result = self.fit(d_train, d_valid, params)
            cv_score_list.append(dict(model.best_score))
            best_iteration += model.best_iteration / n_folds

            # get feature importances
            importances_tmp = pd.DataFrame(
                model.feature_importance("gain"),
                columns=[f'gain_{i_fold+1}'],
                index=x_train.columns
            )
            importances = importances.join(importances_tmp, how='inner')
            models.append(model)

        # predict out-of-fold and test
        n_iteration = int(1.2 * best_iteration)
        oof_preds = model.predict(x_train, num_iteration=n_iteration)
        sub_preds = model.predict(x_test, num_iteration=n_iteration)

        best_thr = 0.5
        oof_preds_tmp = (oof_preds >= best_thr).astype(int)
        best_metric = matthews_corrcoef(y_train, oof_preds_tmp)
        print(f'best_metric: {best_metric}, best_thr: {best_thr}')

        feature_importance = importances.mean(axis=1)
        feature_importance = feature_importance.sort_values(ascending=False).to_dict()

        train_results = {"evals_result": {
            "oof_score": best_metric,
            "best_thr": best_thr,
            "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(cv_score_list)},
            "n_data": len(x_train),
            "best_iteration": best_iteration,
            "n_features": len(x_train.columns),
            "feature_importance": feature_importance
        }}

        return models, oof_preds, sub_preds, train_results
