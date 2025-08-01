"""
FastAPI Service for Property Value Prediction

This API loads a pre-trained machine learning pipeline and exposes
an endpoint to predict house value based on property features.

Endpoints:
- GET /       : Simple health check endpoint.
- POST /predict : Accepts property features as JSON, returns predicted price.

Author: Tobias Holmes
Date: 2025-07
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import os
import joblib

# Init API instance
app = FastAPI()

# Define project root and model path dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(PROJECT_ROOT, "models", "pipeline.joblib")

# Load the pre-trained pipeline model
model = joblib.load(model_path)

# Pydantic input definitions
class PropertyFeatures(BaseModel):
    """
    Pydantic model defining input schema for property features.
    
    Fields correspond to the dataset columns used for prediction.
    Optional fields allow missing values for some features.
    
    Uses Field with alias for fields starting with digits.
    """
    MSSubClass: int
    MSZoning: str
    LotFrontage: Optional[float] = None
    LotArea: int
    Street: str
    Alley: Optional[str] = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: Optional[float] = None
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: Optional[float] = None
    BsmtUnfSF: Optional[float] = None
    TotalBsmtSF: Optional[float] = None
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    # Fields starting with digits need alias, e.g., "1stFlrSF"
    firstFlrSF: Optional[int]  = Field(...,alias="1stFlrSF") # can't start with digit, add underscore
    secondFlrSF: Optional[int] = Field(...,alias="2ndFlrSF")
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[int] = None
    GarageFinish: Optional[str] = None
    GarageCars: Optional[int] = None
    GarageArea: Optional[int] = None
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    threeSsnPorch: Optional[int] = Field(...,alias="3SsnPorch")
    ScreenPorch: int
    PoolArea: int
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

    model_config = {
        "populate_by_name": True,  # Allow population by alias or attribute name
        "from_attributes": True,   # Allow creation from arbitrary objects with attributes
    }


@app.get("/")
def read_root() -> dict:
    """
    Health check endpoint.

    Returns:
    -------
    dict
        Status message confirming the service is running.
    """
    return {"message" : "Property Value Predictor is running"}

@app.post("/predict")
def predict(data: PropertyFeatures, decimals: int = 2) -> dict:
    """
    Endpoint to predict the sale value of a property.

    Parameters:
    ----------
    data : PropertyFeatures
        Validated input property features.
    decimals : int, optional
        Number of decimal places for prediction. Defaults to 2.

    Returns:
    -------
    dict
        Predicted value rounded to the specified decimals.
    """
    
    # Convert input data to DataFrame for the pipeline
    df = pd.DataFrame([data.model_dump(by_alias=True)])

    # Use loaded model to predict value
    prediction = model.predict(df)

    return {"predicted price": round(prediction[0], decimals)}