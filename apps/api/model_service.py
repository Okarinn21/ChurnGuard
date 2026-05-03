import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import HTTPException
from typing import Dict, Any, List

from apps.api.schemas import (
    CustomerInput, PredictionResponse, BatchPredictionResponse,
    ModelInfoResponse, FeatureFactor
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "dataset"

FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

FEATURE_LABELS = {
    "CreditScore": "Điểm tín dụng",
    "Geography": "Quốc gia",
    "Gender": "Giới tính",
    "Age": "Tuổi",
    "Tenure": "Số năm gắn bó",
    "Balance": "Số dư TK",
    "NumOfProducts": "Số sản phẩm",
    "HasCrCard": "Có thẻ TD",
    "IsActiveMember": "Thành viên hoạt động",
    "EstimatedSalary": "Thu nhập ước tính",
}


def risk_level(prob: float) -> str:
    if prob >= 0.6:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    return "Low"


def make_recommendation(prob: float, top_factors: List[FeatureFactor]) -> str:
    if prob < 0.35:
        return "Khách hàng ổn định. Duy trì chăm sóc định kỳ."
    elif prob < 0.6:
        top_name = top_factors[0].feature if top_factors else ""
        return (
            f"Nguy cơ trung bình. Chú ý yếu tố '{FEATURE_LABELS.get(top_name, top_name)}'. "
            "Cân nhắc ưu đãi giữ chân phù hợp."
        )
    else:
        return (
            "Nguy cơ cao! Cần liên hệ chủ động ngay. "
            "Đề xuất gói ưu đãi đặc biệt hoặc tư vấn viên cá nhân."
        )


class ModelService:
    def __init__(self):
        self._start_time = time.time()
        self.model_name = "XGBoost"
        self._model = None
        self._le_geo = None
        self._le_gen = None
        self._threshold: float = 0.5         
        self._meta: Dict[str, Any] = {}
        self._feature_importances: Dict[str, float] = {}
        self._dataset_stats: Dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            self._model  = joblib.load(MODEL_DIR / "xgboost.pkl")
            self._le_geo = joblib.load(MODEL_DIR / "le_geography.pkl")
            self._le_gen = joblib.load(MODEL_DIR / "le_gender.pkl")

            with open(MODEL_DIR / "meta.json") as f:
                self._meta = json.load(f)

            self._threshold = float(self._meta.get("best_threshold", 0.5))

            importances = self._model.feature_importances_
            self._feature_importances = {
                feat: float(imp)
                for feat, imp in zip(FEATURE_COLS, importances)
            }

            self._load_dataset_stats()
            print(f"[ModelService]  XGBoost loaded | threshold={self._threshold}")
        except Exception as e:
            print(f"[ModelService]  Load error: {e}")

    def _load_dataset_stats(self):
        try:
            df = pd.read_csv(DATA_DIR / "train.csv")
            total = len(df)
            churn = int(df["Exited"].sum())
            self._dataset_stats = {
                "total_samples": total,
                "churn_count": churn,
                "stay_count": total - churn,
                "churn_rate": round(churn / total, 4),
                "features_summary": {
                    col: {
                        "mean": round(float(df[col].mean()), 2) if pd.api.types.is_numeric_dtype(df[col]) else None,
                        "std":  round(float(df[col].std()),  2) if pd.api.types.is_numeric_dtype(df[col]) else None,
                        "min":  round(float(df[col].min()),  2) if pd.api.types.is_numeric_dtype(df[col]) else None,
                        "max":  round(float(df[col].max()),  2) if pd.api.types.is_numeric_dtype(df[col]) else None,
                        "top_values": df[col].value_counts().head(3).to_dict() if not pd.api.types.is_numeric_dtype(df[col]) else None,
                    }
                    for col in FEATURE_COLS
                },
            }
        except Exception as e:
            print(f"[ModelService] Stats load error: {e}")

    def is_loaded(self) -> bool:
        return self._model is not None

    def uptime(self) -> float:
        return round(time.time() - self._start_time, 2)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[FEATURE_COLS].copy()
        df["Geography"] = self._le_geo.transform(df["Geography"])
        df["Gender"]    = self._le_gen.transform(df["Gender"])
        return df

    def predict_single(self, customer: CustomerInput) -> PredictionResponse:
        if not self.is_loaded():
            raise HTTPException(status_code=503, detail="Model chưa được tải")

        row = pd.DataFrame([customer.model_dump()])
        X = self._preprocess(row)
        proba = self._model.predict_proba(X)[0]
        prob_churn = float(proba[1])

        prediction = int(prob_churn >= self._threshold)

        top_factors = sorted(
            [
                FeatureFactor(
                    feature=feat,
                    importance=round(imp, 4),
                    value=customer.model_dump()[feat],
                )
                for feat, imp in self._feature_importances.items()
            ],
            key=lambda x: x.importance,
            reverse=True,
        )[:5]

        rec = make_recommendation(prob_churn, top_factors)

        return PredictionResponse(
            prediction=prediction,
            churn_probability=round(prob_churn, 4),
            risk_level=risk_level(prob_churn),
            top_factors=top_factors,
            recommendation=rec,
        )

    def _validate_batch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Thiếu các cột: {missing}. Cần: {FEATURE_COLS}"
            )
        return df

    def predict_batch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_loaded():
            raise HTTPException(status_code=503, detail="Model chưa được tải")

        df = self._validate_batch_df(df)
        result = df.copy()
        X = self._preprocess(df)
        proba = self._model.predict_proba(X)
        result["churn_probability"] = np.round(proba[:, 1], 4)

        result["prediction"] = (proba[:, 1] >= self._threshold).astype(int)
        result["risk_level"] = result["churn_probability"].apply(risk_level)
        return result

    def predict_batch(self, df: pd.DataFrame) -> BatchPredictionResponse:
        result = self.predict_batch_df(df)
        predictions = result.to_dict(orient="records")

        churn_count = int((result["prediction"] == 1).sum())
        total       = len(result)
        high_risk   = int((result["risk_level"] == "High").sum())
        medium_risk = int((result["risk_level"] == "Medium").sum())
        low_risk    = int((result["risk_level"] == "Low").sum())

        return BatchPredictionResponse(
            total_customers=total,
            churn_count=churn_count,
            stay_count=total - churn_count,
            churn_rate=round(churn_count / total, 4),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            predictions=predictions,
        )

    def get_info(self) -> ModelInfoResponse:
        stats = self._dataset_stats

        xgb_metrics = self._meta.get("val_metrics", {}).get("xgboost", {})

        return ModelInfoResponse(
            model_name=self.model_name,
            algorithm="XGBoost Classifier",
            n_features=len(FEATURE_COLS),
            features=FEATURE_COLS,
            training_samples=stats.get("total_samples", 0),
            train_auc=xgb_metrics.get("auc", 0),
            geography_classes=self._meta.get("geography_classes", []),
            gender_classes=self._meta.get("gender_classes", []),
        )

    def get_feature_importance(self) -> Dict[str, Any]:
        sorted_fi = sorted(
            self._feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        return {
            "model": self.model_name,
            "threshold": self._threshold,
            "feature_importance": [
                {"feature": k, "label": FEATURE_LABELS.get(k, k), "importance": round(v, 4)}
                for k, v in sorted_fi
            ]
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        return self._dataset_stats