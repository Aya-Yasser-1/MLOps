import os
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def read_process_data(
    cfg: DictConfig,
    logger,
    ) -> None:
    logger.info("Data Processing started")

    df = pd.read_csv(os.path.join(cfg.pipeline.data.source_dir, f"{cfg.pipeline.data.file_name}.csv"))
    df.set_index(cfg.pipeline.data.id_col, inplace=True)

    df.drop(columns=cfg.pipeline.preprocessing.drop_columns, inplace=True)

    df[cfg.pipeline.preprocessing.cat_fill] = df[cfg.pipeline.preprocessing.cat_fill].fillna(
        df[cfg.pipeline.preprocessing.cat_fill].mode().iloc[0]
    )

    df[cfg.pipeline.preprocessing.numeric_fill] = df[cfg.pipeline.preprocessing.numeric_fill].fillna(
        df[cfg.pipeline.preprocessing.numeric_fill].mean()
    )

    train_df, test_df = train_test_split(
        df, test_size=cfg.pipeline.preprocessing.test_size,
        random_state=cfg.pipeline.preprocessing.random_state,
        stratify=df[cfg.pipeline.data.target_col]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cfg.pipeline.preprocessing.numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.pipeline.preprocessing.cat_cols)
        ]
    )

    preprocessor.fit(train_df)

    df_train_processed = preprocessor.transform(train_df)
    df_test_processed = preprocessor.transform(test_df)

    feature_names = preprocessor.get_feature_names_out()

    df_train_processed = pd.DataFrame(df_train_processed, columns=feature_names, index=train_df.index)
    df_test_processed = pd.DataFrame(df_test_processed, columns=feature_names, index=test_df.index)

    df_train_processed[cfg.pipeline.data.target_col] = train_df[cfg.pipeline.data.target_col].values
    df_test_processed[cfg.pipeline.data.target_col] = test_df[cfg.pipeline.data.target_col].values

    joblib.dump(preprocessor, cfg.pipeline.model.preprocessor_path)

    df_train_processed.to_parquet(
        os.path.join(cfg.pipeline.data.processed_dir, f"{cfg.pipeline.data.file_name}_train_processed.parquet")
    )
    df_test_processed.to_parquet(
        os.path.join(cfg.pipeline.data.processed_dir, f"{cfg.pipeline.data.file_name}_test_processed.parquet")
    )
