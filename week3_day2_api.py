from contextlib import asynccontextmanager
from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

# Available for models that reference the Week 3 Day 1 module.
import production_pipeline  # noqa: F401


PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_FOLDER / "final_production_candidate.joblib"

NUMERICAL_FEATURES = [
    "production_rate_per_hour",
    "good_units",
    "units_produced",
    "shift_start_hour",
    "estimated_oee_pct",
    "downtime_minutes",
    "runtime_hours",
    "defect_rate_pct",
    "experience_runtime_interaction",
    "temperature_deviation",
    "defect_count",
    "experience_level",
    "cycle_time_avg",
    "machine_prior_avg_downtime",
    "machine_prior_avg_defect_rate",
    "machine_prior_avg_units",
    "maintenance_downtime",
    "humidity",
]

CATEGORICAL_FEATURES = [
    "shift_name",
    "machine_id",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

model = None
required_features = None


class ManufacturingRecord(BaseModel):
    """One manufacturing record containing all model predictors."""

    model_config = ConfigDict(extra="forbid")

    production_rate_per_hour: float
    good_units: float
    units_produced: float
    shift_start_hour: float
    estimated_oee_pct: float
    downtime_minutes: float
    runtime_hours: float
    defect_rate_pct: float
    experience_runtime_interaction: float
    temperature_deviation: float
    defect_count: float
    shift_name: str
    experience_level: float
    cycle_time_avg: float
    machine_prior_avg_downtime: float
    machine_prior_avg_defect_rate: float
    machine_id: str
    machine_prior_avg_units: float
    maintenance_downtime: float
    humidity: float

    @field_validator(*NUMERICAL_FEATURES, mode="before")
    @classmethod
    def validate_numeric_feature(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("A numerical predictor must be a number")

        if not math.isfinite(value):
            raise ValueError("A numerical predictor must be finite")

        return value

    @field_validator(*CATEGORICAL_FEATURES)
    @classmethod
    def validate_categorical_feature(cls, value):
        if not value.strip():
            raise ValueError("A categorical predictor cannot be empty")

        return value


class PredictionRequest(BaseModel):
    record: ManufacturingRecord


class PredictionResponse(BaseModel):
    predicted_shift_efficiency_score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved pipeline once when the API starts."""
    global model, required_features

    if not MODEL_FILE.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)

    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "The saved model does not expose feature_names_in_."
        )

    required_features = list(model.feature_names_in_)

    if set(required_features) != set(ALL_FEATURES):
        raise RuntimeError(
            "The API schema does not match the saved model. "
            f"Model predictors: {required_features}"
        )

    yield

    model = None
    required_features = None


app = FastAPI(
    title="NorDex Shift Efficiency Prediction API",
    description=(
        "Week 3 Day 2 API for predicting manufacturing "
        "shift efficiency scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Report whether the prediction model is loaded."""
    return {
        "status": "ready" if model is not None else "unavailable",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Validate one record and return its predicted efficiency score."""
    if model is None or required_features is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable",
        )

    record = request.record.model_dump()

    # Match the feature order expected by the saved pipeline.
    input_data = pd.DataFrame(
        [record],
        columns=required_features,
    )

    try:
        prediction = float(model.predict(input_data)[0])
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    if not math.isfinite(prediction):
        raise HTTPException(
            status_code=500,
            detail="The model returned an invalid prediction",
        )

    return PredictionResponse(
        predicted_shift_efficiency_score=prediction
    )
