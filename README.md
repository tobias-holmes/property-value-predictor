[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Property Value Predictor

This project implements a machine learning pipeline to predict house sale prices based on structured property data. It includes data preprocessing, model training with PCA and linear regression, and a dockerised API for serving predictions.

## Features

* **Data preprocessing:** Handling missing values by imputation, one-hot encoding and feature scaling (Standard or MinMax).
* **Model training:** Uses a scikit-learn pipeline with PCA for dimensionality reduction and linear regression.
* **Model evalutation:** Supports train/test split evaluation and 5-fold-cross validation.
* **Model persistance:** Saves trained pipeline and components for reuse. 
* **API inference:** FastAPI web service for price prediction.
* **Testing:** Pytest test suite for data preprocessing validation.

## Project Structure
```bash
.
├── data/                      
│   ├── data.csv               # Main dataset
│   └── description.txt        # Dataset description

├── models/                    # Saved models and pipeline artifacts
│   ├── model.joblib           # Trained regression model
│   ├── pca.joblib             # PCA transformer
│   └── pipeline.joblib        # Complete preprocessing and model pipeline

├── notebooks/                 # Jupyter notebooks for exploration and prototyping

├── src/                       
│   ├── app.py                 # FastAPI application for inference
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   └── training.py            # Model training and evaluation

├── tests/                     
│   └── test_data_preprocessing.py  # Pytest suite for preprocessing functions

├── Dockerfile                 # Docker setup for containerised deployment
├── compose.yml                # Docker compose file
├── requirements-dev.txt       # Development dependencies
├── requirements-docker.txt    # Docker environment dependencies
├── setup-dev.sh               # Shell script to set up the development environment
├── pytest.ini                 # Pytest configuration
├── test-payload.json          # Example payload for API testing
├── test-prediction-with-container.sh  # Script to test prediction via Docker
├── README.md                  # Project overview and instructions
└── LICENSE                    # License file
```
## Getting Started

### Dependencies
* Python 3.8+
* pip
* Docker

### Installation
1. Clone this repository:

```bash
git clone git@github.com:tobias-holmes/property-value-predictor.git
cd property-value-predictor
```
2. Setup development environment (create venv and install requirements-dev.txt):

```bash
bash setup-dev.sh
```

### Useage

#### Training the Model
Run the training script to preprocess data, train the model pipeline, and save artifacts:

```bash
python src/train.py
```
#### Run the API Server
Start the FastAPI server for inference:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8080  
```
API will be available at http://localhost:8000.
* Health check endpoint: GET /
* Prediction endpoint: POST /predict

Example JSON payload for prediction:
```json
{
  "MSSubClass": 60,
  "MSZoning": "RL",
  "LotFrontage": 65.0,
  ...
  "SaleCondition": "Normal"
}

```
Also see `test-payload` for the JSON structure.

#### Test Inference

To test the inference API run:
```bash
bash test-prediction-with-container.sh
```
This returns the predicted value of the property in `test-payload.json`.

#### Run Tests

Run preprocessing tests with:
```bash
pytest
```

## Code Overview

`src/data_preprocessing.py`: Contains functions to clean data, impute missing values, encode categorical variables, and build preprocessing pipelines.

`src/train.py`: Loads data, splits train/test, trains pipeline with PCA and linear regression, evaluates, and saves models.

`app.py`: FastAPI app that loads saved pipeline and provides REST endpoints for prediction.

`tests/test_preprocessing.py`: Unit tests for preprocessing functions using synthetic and real data.

## Notes

* Preprocessing pipeline supports choice between standard and min-max scaling.
* PCA components are selected to explain 95% variance by default.
* The project assumes data file at `data/data.csv`.
* Large integration tests are marked with `@pytest.mark.slow`.

## Outlook

* The cross-validation shows an obviously weak fold and some empty columns which suggests that there are sparsey filled collumns that are potentially not correctly addressed by the preprocessing pipeline due to the train-test split.

* The inference service currently only accepts a single set of features as input. This could be expanded to allow bulk inference.

* The uvicorn service is currently still running single threaded.

* Setting up an async service would allow for better scaling.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Author
Tobias Holmes