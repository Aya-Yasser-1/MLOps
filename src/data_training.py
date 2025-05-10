import os
import dvc.api

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.logger import ExecutorLogger


def trainer(config, logger):

    source_path = config['data']['processed_dir']
    file_name = config['data']['file_name']
    target_col = config['data']['target_col']

    train_df = pd.read_parquet(os.path.join(source_path, f"{file_name}_train_processed.parquet"))
    test_df = pd.read_parquet(os.path.join(source_path, f"{file_name}_test_processed.parquet"))

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    pd.DataFrame(X_test).to_parquet(os.path.join(config['test'], "X_test.parquet"))
    pd.DataFrame(y_test).to_parquet(os.path.join(config['test'], "y_test.parquet"))

    logger.info("Target variable extracted successfully")
    logger.info(f"Number of classes: {len(y_train.unique())}")

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    model_path = config['model']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    logger.info("Model trained and saved successfully")
    return model


if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    cfg = dvc.api.params_show()
    logger.info(
        "Paramsters: \n"
        f"{cfg['model']}"
    )
    trainer(cfg, logger)

