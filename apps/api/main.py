from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import io
import json
import time
from typing import Optional
from contextlib import asynccontextmanager

from apps.api.schemas import (
    CustomerInput, PredictionResponse,
    BatchPredictionResponse, ModelInfoResponse,
    HealthResponse, CustomerRecord
)
from apps.api.model_service import ModelService

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService()
    yield

app = FastAPI(
    title="Bank Churn Prediction API",
    description="API for classifying customers likely to leave the bank",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {"message": "Bank Churn Prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    svc: ModelService = app.state.model_service
    return HealthResponse(
        status="ok",
        model_loaded=svc.is_loaded(),
        model_name=svc.model_name,
        uptime_seconds=svc.uptime(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Infomation about the Loaded Model"""
    svc: ModelService = app.state.model_service
    return svc.get_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerInput):
    svc: ModelService = app.state.model_service
    return svc.predict_single(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file CSV")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc CSV: {e}")

    svc: ModelService = app.state.model_service
    return svc.predict_batch(df)


@app.post("/predict/batch/download", tags=["Prediction"])
async def predict_batch_download(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file CSV")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc CSV: {e}")

    svc: ModelService = app.state.model_service
    result_df = svc.predict_batch_df(df)

    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )


@app.get("/stats/dataset", tags=["Analytics"])
def dataset_stats():
    svc: ModelService = app.state.model_service
    return svc.get_dataset_stats()


@app.get("/stats/feature-importance", tags=["Analytics"])
def feature_importance():
    svc: ModelService = app.state.model_service
    return svc.get_feature_importance()
