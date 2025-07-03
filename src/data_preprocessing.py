import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def impute_missing_values(df):
    """
    Impute missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame with missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    # replace None with np.nan for consistency
    df = df.replace({None: np.nan})

    # Create a copy of the DataFrame to avoid modifying the original
    df_imputed = df.copy(deep=True)

    # Separate columns by type
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Imputer for numerical columns
    numeric_imputer = SimpleImputer(strategy="mean")
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
    categorical_imputer = SimpleImputer(strategy="constant", fill_value="NA")
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


if __name__ == "__main__":
    import os

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

    # Remove Misc columns
    print("🗑️ Removing unnecessary columns (ID, MiscFeature, MiscVal)...")
    df.drop(columns=["Id", "MiscFeature", "MiscVal"], inplace=True)

    # Imputing
    print("🧹 Imputation data...")
    df_imputed = impute_missing_values(df)

    # One-hot encoding
    print("🔠 One-hot encoding categorical variables...")
    df_encoded = one_hot_encoding(df_imputed)

    print("✅ Data preprocessing complete. Saving the processed data...")
    df_encoded.to_csv(
        os.path.join(repo_base, "data", "data_processed.csv"),
        index=False,
    )

    print("📂 Processed data saved successfully. Done :-)")
