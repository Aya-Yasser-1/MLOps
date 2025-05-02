import json
import os
import pickle
import joblib


from sklearn.metrics import classification_report


def evaluate(X_test, y_test, config, logger):
    model_path = config['model']['model_path']
    report_path = config['report']['report_path']

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