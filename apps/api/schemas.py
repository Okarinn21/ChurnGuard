from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, example=650, description="Điểm tín dụng (300–850)")
    Geography: str = Field(..., example="France", description="Quốc gia: France | Germany | Spain")
    Gender: str = Field(..., example="Male", description="Giới tính: Male | Female")
    Age: float = Field(..., ge=18, le=100, example=35, description="Tuổi khách hàng")
    Tenure: int = Field(..., ge=0, le=10, example=5, description="Số năm là khách hàng (0–10)")
    Balance: float = Field(..., ge=0, example=85000.0, description="Số dư tài khoản")
    NumOfProducts: int = Field(..., ge=1, le=4, example=2, description="Số sản phẩm đang sử dụng (1–4)")
    HasCrCard: int = Field(..., ge=0, le=1, example=1, description="Có thẻ tín dụng: 1=Có, 0=Không")
    IsActiveMember: int = Field(..., ge=0, le=1, example=1, description="Thành viên hoạt động: 1=Có, 0=Không")
    EstimatedSalary: float = Field(..., ge=0, example=75000.0, description="Thu nhập ước tính")

    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, v):
        allowed = {"France", "Germany", "Spain"}
        if v not in allowed:
            raise ValueError(f"Geography phải là một trong: {allowed}")
        return v

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"Gender phải là một trong: {allowed}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 5,
                "Balance": 85000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0,
            }
        }
    }


class FeatureFactor(BaseModel):
    feature: str
    importance: float
    value: Any


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Ở lại, 1 = Rời bỏ")
    churn_probability: float = Field(..., description="Xác suất rời bỏ (0.0 – 1.0)")
    risk_level: str = Field(..., description="Low | Medium | High")
    top_factors: List[FeatureFactor] = Field(..., description="Top yếu tố ảnh hưởng")
    recommendation: str = Field(..., description="Gợi ý hành động")


class CustomerRecord(BaseModel):
    id: Optional[int] = None
    CreditScore: int
    Geography: str
    Gender: str
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    prediction: int
    churn_probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    total_customers: int
    churn_count: int
    stay_count: int
    churn_rate: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    n_features: int
    features: List[str]
    training_samples: int
    train_auc: float
    geography_classes: List[str]
    gender_classes: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    uptime_seconds: float
