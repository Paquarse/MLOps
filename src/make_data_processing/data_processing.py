import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from conf import config


def load_data():
    return pd.read_csv(config["data"]["train"])


def split_data(df) :
    X = df.drop(columns=config["data"]["target"])
    y = df[[config["data"]["target"]]]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train, X_valid, y_train, y_valid


def data_processing(df_train, df_valid) :
    # list for cols to scale
    cols_to_scale = config["data"]["numerical"]

    # create and fit scaler
    scaler = StandardScaler()
    scaler.fit(df_train[cols_to_scale])

    path_output_scaler = (
        config["data"]["data_processing_files"] + "train_numerical_scaler.pkl"
    )

    with open(path_output_scaler, "wb") as f:
        pickle.dump(scaler, f)

    # scale selected data
    df_train[cols_to_scale] = scaler.transform(df_train[cols_to_scale])
    df_valid[cols_to_scale] = scaler.transform(df_valid[cols_to_scale])

    return df_train, df_valid


def save_data_processing( X_train, X_valid, y_train, y_valid) :
    path_output_x_train = config["data"]["data_processing_files"] + "x_train.pkl"
    path_output_x_valid = config["data"]["data_processing_files"] + "x_valid.pkl"
    path_output_y_train = config["data"]["data_processing_files"] + "y_train.pkl"
    path_output_y_valid = config["data"]["data_processing_files"] + "y_valid.pkl"

    with open(path_output_x_train, "wb") as f:
        pickle.dump(X_train, f)

    with open(path_output_x_valid, "wb") as f:
        pickle.dump(X_valid, f)

    with open(path_output_y_train, "wb") as f:
        pickle.dump(y_train, f)

    with open(path_output_y_valid, "wb") as f:
        pickle.dump(y_valid, f)


def run_data_processing():
    data = load_data()
    X_train, X_valid, y_train, y_valid = split_data(data)
    X_train, X_valid = data_processing(X_train, X_valid)
    save_data_processing(X_train, X_valid, y_train, y_valid)
