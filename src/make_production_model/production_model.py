import mlflow
from mlflow.tracking import MlflowClient

from conf import config


def run_production_model():
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    runs = mlflow.search_runs(experiment_ids=[int(config["mlflow"]["experiment_id"])])
    max_accuracy = max(
        runs["metrics." + config["mlflow"]["metric_selection_production"]]
    )
    max_accuracy_run_id = list(
        runs[
            runs["metrics." + config["mlflow"]["metric_selection_production"]]
            == max_accuracy
        ]["run_id"]
    )[0]
    result = mlflow.register_model(
        f"runs:/{max_accuracy_run_id}/artifacts/model",
        config["mlflow"]["register_model_name"],
    )

    client = MlflowClient()

    for mv in client.search_model_versions(
        f"name='{config['mlflow']['register_model_name']}'"
    ):
        mv = dict(mv)
        if mv["current_stage"] == "Production":
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=config["mlflow"]["register_model_name"],
                version=current_version,
                stage="Staging",
            )

    client.transition_model_version_stage(
        name=config["mlflow"]["register_model_name"],
        version=result.version,
        stage="Production",
    )
