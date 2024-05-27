config = {
    "data": {
        "train": "/Users/jed/Desktop/insa/data/train.csv.xls",
        "test": "/Users/jed/Desktop/insa/data/test.csv.xls",
        "target": "price_range",
        "numerical": [
            "battery_power",
            "clock_speed",
            "fc",
            "int_memory",
            "m_dep",
            "mobile_wt",
            "pc",
            "px_height",
            "px_width",
            "ram",
            "sc_h",
            "sc_w",
            "talk_time",
        ],
        "data_processing_files": "make_data_processing/files/",
        "predictions_path": "/Users/jed/Desktop/insa/src/make_predictions/files/",
    },
    "mlflow": {
        "tracking_uri": "http://127.0.0.1:5000/",
        "experiment_name": "mobile_price",
        "experiment_id": "1",
        "metric_selection_production": "valid_accuracy_score",
        "register_model_name": "model_mobile_price_prediction",
    },
    "train": {
        "models": ["xgboost", "random_forest", "decision_tree"],
    },
    "xgboost_params": {
        "n_estimators": "150",
        "learning_rate": "0.1",
    },
    "random_forest_params": {
        "n_estimators": "150",
    },
    "decision_tree_params": {
        "splitter": "best",
        "min_samples_split": "2",
    },
} 
