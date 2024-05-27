import pickle

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from conf import config


def get_test_data() :
    return pd.read_csv(config["data"]["test"])


def make_data_processing(X_test) :
    if config["data"]["target"] in X_test.columns:
        X_test = X_test.drop(columns=config["data"]["target"])
    if "id" in X_test.columns:
        X_test = X_test.drop(columns="id")

    # list for cols to scale
    cols_to_scale = config["data"]["numerical"]

    path_output_scaler = (
        config["data"]["data_processing_files"] + "train_numerical_scaler.pkl"
    )

    scaler = pickle.load(open(path_output_scaler, "rb"))

    # scale selected data
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_test


def make_predictions(X_test) :
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_name = config["mlflow"]["register_model_name"]
    stage = "Production"

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == stage:
            model = mlflow.pyfunc.load_model(f"runs:/{mv.run_id}/model")

    return pd.DataFrame(model.predict(X_test), columns=["price_range"])


def save_predictions(predictions) :
    path_output_predictions = (
        config["data"]["predictions_path"] + "test_predictions.csv"
    )
    predictions.to_csv(path_output_predictions, index=False)


def run_make_predictions():
    X_test = get_test_data()
    X_test = make_data_processing(X_test)
    y_predictions = make_predictions(X_test)
    save_predictions(y_predictions)
