import json
import os
import pickle
import joblib
import dvc.api
import pandas as pd

from sklearn.metrics import classification_report
from src.logger import ExecutorLogger


def evaluate(config, logger):
    model_path = config['model']['model_path']
    report_path = config['report']['report_path']

    X_test = pd.read_parquet(os.path.join(config['test'], f"X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(config['test'], f"y_test.parquet"))

    # Load the trained model
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Generate and print the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Evaluation completed. Here is the classification report:")
    print(classification_report(y_test, y_pred))

    # Save report as JSON
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    cfg = dvc.api.params_show()
    logger.info(
        "Paramsters: \n"
        f"{cfg['evaluate']}"
    )
    evaluate(cfg, logger)