from src.logger import ExecutorLogger
from src.model_evaluate import evaluate
from src.preprocess_data import read_process_data
from src.data_training import extract_target, trainer

def main(logger) -> None:
    logger.info("Training started")
    read_process_data("Titanic", "PassengerId", "Survived", logger)
    X, y, X_test, y_test = extract_target("Titanic", "Survived", logger)
    trainer(X, y,logger)
    evaluate(X_test, y_test, "models/random_forest_model.pkl")
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)

#https://github.com/AyaYasser1112002/Not-Configured_MLOps.git