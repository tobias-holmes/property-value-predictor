import numpy as np
import pandas as pd
from src.data_preprocessing import (
    impute_missing_values,
    one_hot_encoding,
    drop_id_misc_columns,
)

# Deactivate silent downcasting warnings
pd.set_option("future.no_silent_downcasting", True)

def test_impute_missing_values():
    # Create a sample DataFrame with missing values
    data = {
        "A": [1, 2, None, 4],
        "B": ["cat", None, "dog", np.nan],
        "C": [None, 2.5, 3.5, np.nan],
        "D": [None, None, None, None],  # All values missing
    }
    df = pd.DataFrame(data)

    # Impute missing values
    df_imputed = impute_missing_values(df)

    # Check if missing values are imputed correctly
    assert df_imputed["A"].isnull().sum() == 0
    assert df_imputed["B"].isnull().sum() == 0
    assert df_imputed["C"].isnull().sum() == 0
    assert df_imputed["D"].isnull().sum() == 0


def test_one_hot_encoding():
    # Create a sample DataFrame with categorical data
    data = {"A": ["cat", "mouse", "dog", "horse"], "B": [1, 2, 3, 4]}
    df = pd.DataFrame(data)

    # Perform one-hot encoding
    df_encoded = one_hot_encoding(df)

    # Check if one-hot encoding is applied correctly
    assert "A_cat" not in df_encoded.columns
    assert "A_dog" in df_encoded.columns
    assert "A_mouse" in df_encoded.columns
    assert (
        df_encoded.shape[1] == 4
    )  # 2 original columns + 2 new one-hot encoded columns (dropping first category)


def test_drop_id_misc_columns():
    # Create sample data
    data = {
        "Id": [1, 2, 3, 4],
        "A": [1, 2, None, 4],
        "B": ["cat", None, "dog", np.nan],
        "C": [None, 2.5, 3.5, np.nan],
        "MiscFeature": ["None", "Shed", None, "Garbage"],
        "MiscVal": [0, 5000, None, 1000],
    }
    df = pd.DataFrame(data)

    # Perform drop
    df_dropped = drop_id_misc_columns(df)

    # Check if the columns are dropped correctly
    assert "Id" not in df_dropped.columns
    assert "MiscFeature" not in df_dropped.columns
    assert "MiscVal" not in df_dropped.columns
    assert df_dropped.shape[1] == 3  # Should have 3 columns left (A, B, C)
