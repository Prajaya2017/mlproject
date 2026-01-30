import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

from src.exception import CustomException

def save_object(file_path, obj):
    """Save a Python object to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a Python object from disk using pickle."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Evaluate multiple models and return R2 scores.

    Args:
        X_train, y_train, X_test, y_test: train/test datasets
        models (dict): {"model_name": model_instance, ...}
        params (dict): {"model_name": param_grid_or_dict, ...}

    Returns:
        report (dict): {"model_name": test_score, ...}
    """
    try:
        report = {}

        for model_name, model in models.items():
            param = params.get(model_name, {})

            # Handle CatBoost separately (GridSearchCV cannot pass lists directly)
            if isinstance(model, CatBoostRegressor):
                # Pick first value from any hyperparameter lists for manual fit
                cat_param = {}
                for k, v in param.items():
                    if isinstance(v, list):
                        cat_param[k] = v[0]
                    else:
                        cat_param[k] = v
                model.set_params(**cat_param)
                model.fit(X_train, y_train, verbose=False)

            else:
                # Use GridSearchCV for sklearn models
                gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Compute R2
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)