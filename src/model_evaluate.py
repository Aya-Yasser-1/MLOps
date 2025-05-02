import json
import os
import pickle
import joblib
from omegaconf import DictConfig

from sklearn.metrics import classification_report


def evaluate(X_test, y_test, cfg: DictConfig, logger):

    # Load the trained model
    model = joblib.load(cfg.model_path)
    y_pred = model.predict(X_test)

    # Generate and print the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Evaluation completed. Here is the classification report:")
    print(classification_report(y_test, y_pred))

    # Save report as JSON
    os.makedirs(os.path.dirname(cfg.report_path), exist_ok=True)
    with open(cfg.report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Classification report saved to {cfg.report_path}")