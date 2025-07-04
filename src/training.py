import pandas as pd
from src.data_preprocessing import preprocessing_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Load data and setting target/label variable
df = pd.read_csv("data/data.csv")
y = df.columns["SalePrice"]
X = df.drop(columns=["SalePrice"])

# Load preprocessing pipeline
preprocessing = preprocessing_pipeline()

# Create full pipeline
pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("pca", PCA(n_components=2)),
        ("model", LinearRegression()),
    ]
)
