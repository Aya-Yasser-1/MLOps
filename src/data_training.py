import os

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def extract_target(
    cfg: DictConfig,
    logger,
):

    train_df = pd.read_parquet(os.path.join(cfg.pipeline.data.processed_dir, f"{cfg.pipeline.data.file_name}_train_processed.parquet"))
    test_df = pd.read_parquet(os.path.join(cfg.pipeline.data.processed_dir, f"{cfg.pipeline.data.file_name}_test_processed.parquet"))

    X_train = train_df.drop(columns=[cfg.pipeline.data.target_col])
    y_train = train_df[cfg.pipeline.data.target_col]
    X_test = test_df.drop(columns=[cfg.pipeline.data.target_col])
    y_test = test_df[cfg.pipeline.data.target_col]

    logger.info("Target variable extracted successfully")
    logger.info(f"Number of classes: {len(y_train.unique())}")

    return X_train, y_train, X_test, y_test

def trainer(X_train, y_train, cfg: DictConfig, logger):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(cfg.pipeline.model.model_path), exist_ok=True)
    joblib.dump(model, cfg.pipeline.model.model_path)

    logger.info("Model trained and saved successfully")
    return model