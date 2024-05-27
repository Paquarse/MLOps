import pickle

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from conf import config


def get_processed_data():
    path_output_x_train = config["data"]["data_processing_files"] + "x_train.pkl"
    path_output_x_valid = config["data"]["data_processing_files"] + "x_valid.pkl"
    path_output_y_train = config["data"]["data_processing_files"] + "y_train.pkl"
    path_output_y_valid = config["data"]["data_processing_files"] + "y_valid.pkl"

    X_train = pickle.load(open(path_output_x_train, "rb"))
    X_valid = pickle.load(open(path_output_x_valid, "rb"))
    y_train = pickle.load(open(path_output_y_train, "rb"))
    y_valid = pickle.load(open(path_output_y_valid, "rb"))

    return X_train, X_valid, y_train, y_valid


def log_metrics(clf, X_train, X_valid, y_train, y_valid):
    # Make predictions with train model

    valid_predictions = clf.predict(X_valid)
    valid_probas = clf.predict_proba(X_valid)

    train_predictions = clf.predict(X_train)
    train_probas = clf.predict_proba(X_train)

    # Train metrics

    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1_score = f1_score(y_train, train_predictions, average="weighted")
    train_precision_score = precision_score(
        y_train, train_predictions, average="weighted"
    )
    train_recall_score = recall_score(y_train, train_predictions, average="weighted")
    train_roc_auc_score = roc_auc_score(
        y_train, train_probas, multi_class="ovr", average="weighted"
    )

    mlflow.log_metric("train_accuracy_score", train_accuracy)
    mlflow.log_metric("train_f1_score", train_f1_score)
    mlflow.log_metric("train_precision_score", train_precision_score)
    mlflow.log_metric("train_recall_score", train_recall_score)
    mlflow.log_metric("train_roc_auc_score", train_roc_auc_score)

    train_confusion_matrix = confusion_matrix(y_train, train_predictions)
    f = sns.heatmap(train_confusion_matrix, annot=True, fmt="d")
    plt.savefig("train_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("train_confusion_matrix.png")

    # Valid metrics

    valid_accuracy = accuracy_score(y_valid, valid_predictions)
    valid_f1_score = f1_score(y_valid, valid_predictions, average="weighted")
    valid_precision_score = precision_score(
        y_valid, valid_predictions, average="weighted"
    )
    valid_recall_score = recall_score(y_valid, valid_predictions, average="weighted")
    valid_roc_auc_score = roc_auc_score(
        y_valid, valid_probas, multi_class="ovr", average="weighted"
    )

    mlflow.log_metric("valid_accuracy_score", valid_accuracy)
    mlflow.log_metric("valid_f1_score", valid_f1_score)
    mlflow.log_metric("valid_precision_score", valid_precision_score)
    mlflow.log_metric("valid_recall_score", valid_recall_score)
    mlflow.log_metric("valid_roc_auc_score", valid_roc_auc_score)

    valid_confusion_matrix = confusion_matrix(y_valid, valid_predictions)
    f = sns.heatmap(valid_confusion_matrix, annot=True, fmt="d")
    plt.savefig("valid_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("valid_confusion_matrix.png")


def train(X_train, X_valid, y_train, y_valid):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    for model in config["train"]["models"]:
        if model == "random_forest":
            mlflow.sklearn.autolog()
            with mlflow.start_run():
                forest_model = RandomForestClassifier(
                    n_estimators=int(config["random_forest_params"]["n_estimators"]),
                    random_state=0,
                )
                forest_model.fit(X_train, y_train)
                mlflow.log_param(
                    "n_estimators", forest_model.get_params()["n_estimators"]
                )
                mlflow.log_param("criterion", forest_model.get_params()["criterion"])
                mlflow.log_param(
                    "min_samples_leaf", forest_model.get_params()["min_samples_leaf"]
                )
                mlflow.log_param(
                    "min_samples_split", forest_model.get_params()["min_samples_split"]
                )
                mlflow.log_param(
                    "random_state", forest_model.get_params()["random_state"]
                )
                log_metrics(forest_model, X_train, X_valid, y_train, y_valid)
                mlflow.end_run()

        if model == "xgboost":
            mlflow.xgboost.autolog()
            with mlflow.start_run():
                model_xgboost = XGBClassifier(
                    n_estimators=int(config["xgboost_params"]["n_estimators"]),
                    random_state=0,
                    learning_rate=float(config["xgboost_params"]["learning_rate"]),
                )
                model_xgboost.fit(X_train, y_train)
                log_metrics(model_xgboost, X_train, X_valid, y_train, y_valid)
                mlflow.end_run()

        if model == "decision_tree":
            mlflow.sklearn.autolog()
            with mlflow.start_run():
                decision_tree = DecisionTreeClassifier(
                    splitter=config["decision_tree_params"]["splitter"],
                    min_samples_split=int(
                        config["decision_tree_params"]["min_samples_split"]
                    ),
                    random_state=0,
                )
                decision_tree.fit(X_train, y_train)
                log_metrics(decision_tree, X_train, X_valid, y_train, y_valid)
                mlflow.end_run()


def run_train():
    X_train, X_valid, y_train, y_valid = get_processed_data()
    train(X_train, X_valid, y_train, y_valid)
