import os
import dvc.api
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.logger import ExecutorLogger
import joblib

class DataFramePreprocessor:
    def __init__(self, id_col=None, drop_cols=None, cat_fill=None, numeric_fill=None):
        self.id_col = id_col
        self.drop_cols = drop_cols
        self.cat_fill = cat_fill
        self.numeric_fill = numeric_fill
        self.cat_modes_ = None
        self.numeric_means_ = None
        
    def fit(self, X, y=None):
        if self.cat_fill:
            self.cat_modes_ = X[self.cat_fill].mode().iloc[0]
        if self.numeric_fill:
            self.numeric_means_ = X[self.numeric_fill].mean()
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.id_col:
            X.set_index(self.id_col, inplace=True)
        if self.drop_cols:
            X.drop(columns=self.drop_cols, inplace=True)
        if self.cat_fill and self.cat_modes_ is not None:
            X[self.cat_fill] = X[self.cat_fill].fillna(self.cat_modes_)
        if self.numeric_fill and self.numeric_means_ is not None:
            X[self.numeric_fill] = X[self.numeric_fill].fillna(self.numeric_means_)
        return X

def read_process_data(config, logger) -> None:
    logger.info("Data Processing started")

    # Config parameters
    file_name = config['data']['file_name']
    id_col = config['data']['id_col']
    target_col = config['data']['target_col']
    source = config['data']['source_dir']
    drop_cols = config['preprocessing']['drop_columns']
    cat_cols = config['preprocessing']['cat_cols']
    numeric_cols = config['preprocessing']['numeric_cols']
    cat_fill = config['preprocessing']['cat_fill']
    numeric_fill = config['preprocessing']['numeric_fill']
    destination = config['data']['processed_dir']
    pipeline_path = config['model']['pipeline_path']
    
    # Load data
    df = pd.read_csv(os.path.join(source, f"{file_name}.csv"))
    
    # Create full pipeline
    df_preprocessor = DataFramePreprocessor(
        id_col=id_col,
        drop_cols=drop_cols,
        cat_fill=cat_fill,
        numeric_fill=numeric_fill
    )
    
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder='passthrough'
    )
    
    model = RandomForestClassifier(
        n_estimators=config['model']['params']['n_estimators'],
        max_depth=config['model']['params']['max_depth'],
        random_state=config['model']['params']['random_state']
    )
    
    full_pipeline = Pipeline([
        ('dataframe_processing', df_preprocessor),
        ('feature_processing', column_transformer),
        ('model', model)
    ])
    
    # Train-test split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_state'],
        stratify=y
    )
    
    # Fit pipeline
    full_pipeline.fit(X_train, y_train)
    
    # Save complete pipeline
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    joblib.dump(full_pipeline, pipeline_path)
    logger.info(f"Full pipeline saved to {pipeline_path}")
    
    # Transform data using pipeline
    train_processed = full_pipeline[:-1].transform(X_train)  # Exclude model step
    test_processed = full_pipeline[:-1].transform(X_test)
    
    # Get feature names
    feature_names = full_pipeline.named_steps['feature_processing'].get_feature_names_out()
    
    # Save processed data
    df_train_processed = pd.DataFrame(train_processed, columns=feature_names, index=X_train.index)
    df_test_processed = pd.DataFrame(test_processed, columns=feature_names, index=X_test.index)
    
    df_train_processed[target_col] = y_train.values
    df_test_processed[target_col] = y_test.values
    
    os.makedirs(destination, exist_ok=True)
    df_train_processed.to_parquet(os.path.join(destination, f"{file_name}_train_processed.parquet"))
    df_test_processed.to_parquet(os.path.join(destination, f"{file_name}_test_processed.parquet"))
    
    logger.info("Data processing completed successfully")

if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    cfg = dvc.api.params_show()
    read_process_data(cfg, logger)