import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


def drop_id_misc_columns(df):
    """
    Drop unnecessary columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.

    Returns:
    pd.DataFrame: DataFrame with specified columns dropped.
    """
    columns_to_drop = ["Id", "MiscFeature", "MiscVal"]
    return df.drop(columns=columns_to_drop, errors="ignore")


def impute_missing_values(df):
    """
    Impute missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame with missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    # Replace None with np.nan for consistency
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


def one_hot_encoding(df):
    """
    Perform one-hot encoding on categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to encode.

    Returns:
    pd.DataFrame: DataFrame with one-hot encoded categorical columns.
    """
    return pd.get_dummies(df, drop_first=True)


def preprocessing_pipeline():
    return Pipeline(
        [
            ("drop_id_misc", FunctionTransformer(drop_id_misc_columns, validate=False)),
            (
                "impute_missing",
                FunctionTransformer(impute_missing_values, validate=False),
            ),
            ("one_hot_encode", FunctionTransformer(one_hot_encoding, validate=False)),
        ]
    )


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

    print("📂 Processed data saved successfully. Done :-)")
