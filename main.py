import hydra

from src.logger import ExecutorLogger
from omegaconf import DictConfig, OmegaConf
from src.preprocess_data import read_process_data
from src.data_training import extract_target, trainer
from src.model_evaluate import evaluate

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = ExecutorLogger("training")
    logger.info("Training started")
    
    read_process_data(cfg, logger)
    X, y, X_test, y_test = extract_target(cfg, logger)
    trainer(X, y, cfg, logger)
    evaluate(X_test, y_test, cfg, logger)
    
    logger.info("Training finished")

if __name__ == "__main__":
    main()
