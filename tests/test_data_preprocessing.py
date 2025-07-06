"""
Pytest test suite for data preprocessing functions in src.data_preprocessing.

Tests include:
- impute_missing_values: checks correct imputation of missing values in numeric and categorical columns.
- one_hot_encoding: verifies one-hot encoding of categorical variables with drop-first category.
- drop_id_misc_columns: ensures specified columns ('Id', 'MiscFeature', 'MiscVal') are removed properly.
- get_preprocessing_pipeline: tests the full preprocessing pipeline output, validating numeric scaling and encoding.
- test_get_preprocessing_pipeline_with_data: integration test using real dataset CSV, marked as slow.

General:
- Uses pandas, numpy and pytest.
- Suppresses pandas future warnings about silent downcasting.
- Validates output types, missing data handling, and expected transformations.
- Includes a slow integration test that requires external CSV file.

Author: Tobias Holmes
Date: 2025-07
"""

import numpy as np
import pytest
import pandas as pd
from src.data_preprocessing import (
    impute_missing_values,
    one_hot_encoding,
    drop_id_misc_columns,
    get_preprocessing_pipeline,
)

# Deactivate silent downcasting warnings
pd.set_option("future.no_silent_downcasting", True)

def test_impute_missing_values():
    """
    Test that missing values are correctly imputed in numeric and categorical columns.
    Checks that no NaNs remain after imputation.
    """

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
    """"
     Test one-hot encoding on categorical columns.
    Verifies first category is dropped (reference category),
    and expected new columns appear.    
    """

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
    """
    Test dropping of 'Id', 'MiscFeature', and 'MiscVal' columns.
    Verifies those columns are removed and others remain intact.
    """

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

def test_get_preprocessing_pipeline():
    """
    Test the full preprocessing pipeline with numeric and categorical data,
    including imputation, scaling (standard and minmax), and encoding.

    Checks:
    - Output is numpy ndarray
    - No missing values after transform
    - Output dimensions greater than numeric-only input
    - Output dtype is numeric
    - MinMax scaled output lies between 0 and 1
    """

    # Create a sample DataFrame with numeric and categorical columns, and missing values
    data = {
        "Id": [1, 2, 3, 4],
        "A": [1, 2, None, 4],
        "B": ["cat", None, "dog", np.nan],
        "C": [None, 2.5, 3.5, np.nan],
        "MiscFeature": ["None", "Shed", None, "Garbage"],
        "MiscVal": [0, 5000, None, 1000],
    }
    df = pd.DataFrame(data)

    # Get pipelines with different scalers
    pipeline = get_preprocessing_pipeline(df, scaler="standard")
    pipeline_minmax = get_preprocessing_pipeline(df, scaler="minmax")


    # Fit and transform
    X_out = pipeline.fit_transform(df)
    X_out_minmax = pipeline_minmax.fit_transform(df)

    # Check output type
    assert isinstance(X_out, np.ndarray)
    assert isinstance(X_out_minmax, np.ndarray)
    # Check if the output has no missing columns
    assert np.isnan(X_out).sum() == 0
    assert np.isnan(X_out_minmax).sum() == 0
    # Should have more columns than just the numeric ones
    assert X_out.shape[1] > 2
    assert X_out_minmax.shape[1] > 2
    # Check that all columns are numeric (float or int)
    assert np.issubdtype(X_out.dtype, np.number)
    assert np.issubdtype(X_out_minmax.dtype, np.number)
    # Check that minmax scaler output is no larger than 1
    assert np.all((X_out_minmax >= 0) & (X_out_minmax <= 1))

@pytest.mark.slow
def test_get_preprocessing_pipeline_with_data():
    """
    Slow integration test on actual data from CSV file.

    Validates full preprocessing pipeline can fit and transform the real dataset
    with no missing values and expected numeric output.
    """

    # Get data from CSV file
    df = pd.read_csv("data/data.csv")
   
    # Get pipeline
    pipeline = get_preprocessing_pipeline(df)

    # Fit and transform
    X_out = pipeline.fit_transform(df)

    # Check output type
    assert isinstance(X_out, np.ndarray)
    # Check if the output has no missing columns
    assert np.isnan(X_out).sum() == 0
    # Should have more columns than just the numeric ones
    assert X_out.shape[1] > 2
    # Check that all columns are numeric (float or int)
    assert np.issubdtype(X_out.dtype, np.number)