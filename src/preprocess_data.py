import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def read_process_data(
    config,
    logger,
) -> None:
    logger.info("Data Processing started")

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
    preprocessor_path = config['model']['preprocessor_path']

    df = pd.read_csv(os.path.join(source, f"{file_name}.csv"))
    df.set_index(id_col, inplace=True)

    df.drop(columns = drop_cols,inplace = True)

    df[cat_fill] = df[cat_fill].fillna(df[cat_fill].mode())

    df[numeric_fill].fillna(df[numeric_fill].mean(),inplace = True)

    train_df, test_df = train_test_split(
        df, test_size=config['preprocessing']['test_size'], 
        random_state=config['preprocessing']['random_state'], stratify=df[target_col]
    )

    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
    )

    preprocessor.fit(train_df)

    df_train_processed = preprocessor.transform(train_df)
    df_test_processed = preprocessor.transform(test_df)

    feature_names = preprocessor.get_feature_names_out()

    df_train_processed = pd.DataFrame(df_train_processed, columns=feature_names, index=train_df.index)
    df_test_processed = pd.DataFrame(df_test_processed, columns=feature_names, index=test_df.index)

    df_train_processed[target_col] = train_df[target_col].values
    df_test_processed[target_col] = test_df[target_col].values

    joblib.dump(preprocessor, preprocessor_path)


    df_train_processed.to_parquet(os.path.join(destination, f"{file_name}_train_processed.parquet"))
    df_test_processed.to_parquet(os.path.join(destination, f"{file_name}_test_processed.parquet"))