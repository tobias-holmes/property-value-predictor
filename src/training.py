import pandas as pd
import joblib
from data_preprocessing import get_preprocessing_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data and setting target/label variable
df = pd.read_csv("data/data.csv")
y = df["SalePrice"]
X = df.drop(columns=["SalePrice"])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load preprocessing pipeline
preprocessing = get_preprocessing_pipeline(X, scaler="standard") # TODO: passing X here might not be ideal

# Create full pipeline
pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("pca", PCA(n_components=125)), # TODO: Gridsearch for best PCA value
        ("model", LinearRegression(fit_intercept=True)),
    ]
)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model and print the R^2 score
score = pipeline.score(X_test, y_test)
print(f"Model R^2 score: {score:.4f}")

#Save the model
joblib.dump(pipeline, "models/pipeline.joblib")
# joblib.dump(pipeline.named_steps["pca"], "models/model.joblib")
joblib.dump(pipeline.named_steps["model"], "models/model.joblib")