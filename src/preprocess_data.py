import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


SOURCE = os.path.join("data", "raw")
DESTINATION = os.path.join("data", "processed")
Preprocess = os.path.join("models","Preprocessor")


def read_process_data(
    file_name: str,
    id_col: str,
    target_col: str,
    logger,
) -> None:
    logger.info("Data Processing started")
    df = pd.read_csv(os.path.join(SOURCE, f"{file_name}.csv"))
    df.set_index(id_col, inplace=True)


    cat_cols = ['Sex','Embarked']
    numeric_cols = ['Age','Fare']
    
    df.drop(columns = ['Cabin','Name','Ticket'],inplace = True)

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())

    df ['Age']=df['Age'].fillna(df['Age'].mean())

    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df[target_col]
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

    joblib.dump(preprocessor, os.path.join(Preprocess, f"{file_name}_preprocessor.pkl"))

    feature_names = preprocessor.get_feature_names_out()

    df_train_processed = pd.DataFrame(df_train_processed, columns=feature_names, index=train_df.index)
    df_test_processed = pd.DataFrame(df_test_processed, columns=feature_names, index=test_df.index)

    df_train_processed[target_col] = train_df[target_col].values
    df_test_processed[target_col] = test_df[target_col].values

    df_train_processed.to_parquet(
        os.path.join(DESTINATION, f"{file_name}_train_processed.parquet"), engine="pyarrow"
    )
    df_test_processed.to_parquet(
        os.path.join(DESTINATION, f"{file_name}_test_processed.parquet"), engine="pyarrow"
    )