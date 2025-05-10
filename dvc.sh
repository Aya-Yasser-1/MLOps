dvc stage add -n preprocessing \
-d data/raw/Titanic.csv \
-d src/preprocess_data.py \
-o data/processed/Titanic_train_processed.parquet \
-o data/processed/Titanic_test_processed.parquet \
-p data -p preprocessing -p model --force \
uv run src/preprocess_data.py 

dvc stage add -n training \
-d src/data_training.py \
-d data/processed/Titanic_train_processed.parquet \
-d data/processed/Titanic_test_processed.parquet \
-o data/test/X_test.parquet \
-o data/test/y_test.parquet \
-o models/random_forest_model.pkl \
-p data -p preprocessing -p model -p test --force \
uv run src/data_training.py 

dvc stage add -n evaluation \
-d src/model_evaluate.py \
-d models/random_forest_model.pkl \
-d data/test/X_test.parquet \
-d data/test/y_test.parquet \
-p model -p test -p report --force \
uv run src/model_evaluate.py
