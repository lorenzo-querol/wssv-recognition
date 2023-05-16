"""
Baseline sweep configuration
"""
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_f1_score",
        "goal": "maximize",
    },
    "parameters": {
        "optimizer": {"value": "adam"},
        "epochs": {"values": [200, 250, 300, 350, 400]},
        # "es_patience": {"values": [20, 25, 30, 35, 40]},
        "learning_rate": {"distribution": "uniform", "min": 0.00001, "max": 0.1},
        "batch_size": {
            "values": [8, 16, 32, 64, 128],
        },
        "dropout_rate": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
    },
}


"""
Third Sweep
"""
sweep_config = {
    "method": "random",
    "metric": {
        "name": "val_f1_score",
        "goal": "maximize",
    },
    "parameters": {
        "optimizer": {"value": "adam"},
        "epochs": {"values": [100, 200, 300]},
        "learning_rate": {"distribution": "uniform", "min": 1e-4, "max": 1e-3},
        "batch_size": {"values": [4, 8, 16, 32, 64]},
        "dropout_rate": {"distribution": "uniform", "min": 0.3, "max": 0.5},
    },
}
