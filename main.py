from src.logger import ExecutorLogger
from src.preprocess_data import read_process_data
from src.data_training import extract_target, trainer
from src.model_evaluate import evaluate
from src.config_loader import load_config

def main(logger):
    config = load_config()
    logger.info("Training started")
    
    read_process_data(config, logger)
    X, y, X_test, y_test = extract_target(config, logger)
    trainer(X, y, config, logger)
    evaluate(X_test, y_test, config, logger)
    
    logger.info("Training finished")

if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)