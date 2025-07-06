import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns from a DataFrame.
    
    Parameters:
    columns (list): List of column names to drop.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")

def drop_id_misc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.

    Returns:
    pd.DataFrame: DataFrame with specified columns dropped.
    """
    columns_to_drop = ["Id", "MiscFeature", "MiscVal"]
    return df.drop(columns=columns_to_drop, errors="ignore")


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame with missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    # Replace None with np.nan for consistency. Infer objects to ensure correct data types.
    df = df.replace({None: np.nan}).infer_objects(copy=False)

    # Create a copy of the DataFrame to avoid modifying the original
    df_imputed = df.copy(deep=True)

    # Separate columns by type
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Imputer for numerical columns
    numeric_imputer = SimpleImputer(strategy="constant", fill_value=0) #TODO: add keep_empty_feature=True from scikit-learn 1.8 onwards
    if numeric_cols.empty:
        print("No numeric columns to impute.")
    else:
        df_imputed[numeric_cols] = (
            pd.DataFrame(  # DataFrame wrapping to ensure correct shape
                numeric_imputer.fit_transform(df[numeric_cols]),
                columns=numeric_cols,
                index=df.index,
            )
        )

    # Imputer for categorical columns
    categorical_imputer = SimpleImputer(strategy="constant", fill_value="NA") #TODO: add keep_empty_feature=True from scikit-learn 1.8 onwards
    if categorical_cols.empty:
        print("No categorical columns to impute.")
    else:
        df_imputed[categorical_cols] = pd.DataFrame(
            categorical_imputer.fit_transform(df[categorical_cols]),
            columns=categorical_cols,
            index=df.index,
        )

    return df_imputed


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding on categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to encode.

    Returns:
    pd.DataFrame: DataFrame with one-hot encoded categorical columns.
    """
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    # Drop original categorical columns and concat encoded
    df_non_cat = df.drop(columns=categorical_cols)
    return pd.concat([df_non_cat, encoded_df], axis=1)


def get_preprocessing_pipeline(df: pd.DataFrame, scaler: str = "standard") -> Pipeline:
    df_dropped = drop_id_misc_columns(df)
    numeric_cols = df_dropped.select_dtypes(include=["number"]).columns
    categorical_cols = df_dropped.select_dtypes(include=["object"]).columns

    # Select scaler based on input
    if scaler == "standard":
        scaler_step = StandardScaler()
    elif scaler == "minmax":
        scaler_step = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler option: {scaler}")

    # Pipeline for numerical columns: impute then scale
    num_pipeline = Pipeline([
        ("num_impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", scaler_step)
    ])

    # Pipeline for categorical columns: impute then encode
    cat_pipeline = Pipeline([
        ("cat_impute", SimpleImputer(strategy="constant", fill_value="NA")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    # Column transformer to apply num and cat pipeline in parallel
    col_transformer = ColumnTransformer([
        ("num_pipeline", num_pipeline, numeric_cols),
        ("cat_pipeline", cat_pipeline, categorical_cols)
    ])

    return Pipeline([
            ("drop_id_misc", DropColumns(columns=["Id", "MiscFeature", "MiscVal"])),
            ("col_transformer", col_transformer),
    ])


if __name__ == "__main__":
    import os

    print("The file was called directly. Starting data preprocessing.")

    # Get the base directory of the repository
    repo_base = os.popen("git rev-parse --show-toplevel").read().strip()

    # Check if data.csv exists in the data folder
    data_path = os.path.join(repo_base, "data", "data.csv")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    else:
        print(f"✅ Data file found at: {data_path}")

    # Load the data
    print("📥 Loading data...")
    df = pd.read_csv(os.path.join(repo_base, "data", "data.csv"))

    # Remove ID and Misc columns
    print("🗑️ Removing unnecessary columns (ID, MiscFeature, MiscVal)...")
    df_dropped = drop_id_misc_columns(df)

    # Imputing
    print("🧹 Imputation data...")
    df_imputed = impute_missing_values(df_dropped)

    # One-hot encoding
    print("🔠 One-hot encoding categorical variables...")
    df_encoded = one_hot_encoding(df_imputed)

    print("✅ Data preprocessing complete. Saving the processed data...")
    df_encoded.to_csv(
        os.path.join(repo_base, "data", "data_processed.csv"),
        index=False,
    )
    
    # Process data with pipeline
    print("🔄 Processing data with preprocessing pipeline...")
    preprocessing_pipeline = get_preprocessing_pipeline(df)
    df_processed = preprocessing_pipeline.fit_transform(df)
    df_processed = pd.DataFrame(df_processed, columns=preprocessing_pipeline.named_steps['col_transformer'].get_feature_names_out())

    print("✅ Data preprocessing with pipeline complete. Saving the processed data...")
    df_processed.to_csv(
        os.path.join(repo_base, "data", "data_processed_pipeline.csv"),
        index=False,
    )

    print("📂 Processed data saved successfully. Done :-)")
