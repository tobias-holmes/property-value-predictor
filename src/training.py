"""
Model Training and Evaluation Script

This script loads housing data, splits it into training and test sets,
builds a machine learning pipeline including preprocessing, PCA,
and linear regression, then trains and evaluates the model.
The trained pipeline, PCA transformer, and model are saved to disk.

It also performs exploratory K-Fold cross-validation.

Author: Tobias Holmes
Date: 2025-07
"""

import pandas as pd
import joblib
from data_preprocessing import get_preprocessing_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

# Load data
df = pd.read_csv("data/data.csv")

# Serparate features and target variable
y = df["SalePrice"]
X = df.drop(columns=["SalePrice"])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load preprocessing pipeline
preprocessing = get_preprocessing_pipeline(X, scaler="standard") 

# Construct full pipeline
pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("pca", PCA(n_components=0.95)), # TODO: Gridsearch for best PCA value?
        ("model", LinearRegression(fit_intercept=True)),
    ]
)

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data and print the R^2 score
score = pipeline.score(X_test, y_test)
print(f"Model R^2 score: {score:.4f}")

# Save the trained pipeline and components for later use
joblib.dump(pipeline, "models/pipeline.joblib")
joblib.dump(pipeline.named_steps["pca"], "models/pca.joblib")
joblib.dump(pipeline.named_steps["model"], "models/model.joblib")


# Perform 5-fold cross validation to estimate model robustness
kf = KFold(n_splits=5, shuffle=True, random_state=42)
score_kf = cross_val_score(pipeline, X, y , cv=kf, scoring="r2")

print(f"Cross-validation R^2 scores: {score_kf}")
print(f"Mean R^2 score: {score_kf.mean():.4f}")
