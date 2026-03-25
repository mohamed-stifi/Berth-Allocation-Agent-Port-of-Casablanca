"""
Layer 2A — Service Time Predictor (Machine Learning).

Responsibility: given a vessel's features, predict how many hours
it will occupy a berth (service_time_hours).

Design:
  - Baseline  : LinearRegression  (interpretable, fast)
  - Production: XGBoost           (higher accuracy)
  - Interface : ServiceTimePredictor wraps both behind one .predict()
  - Persistence: joblib .pkl — one file per model
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config.constants import (
    COL_SERVICE_HOURS, COL_GROSS_TONNAGE, COL_LENGTH, COL_DRAFT,
    COL_TONNAGE_IMPORT, COL_TONNAGE_EXPORT, COL_TUGS_BERTHING,
    COL_ARRIVAL_MONTH, COL_ARRIVAL_DOW,
    TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE, SERVICE_TIME_MODEL,
)
from config.models import Vessel

logger = logging.getLogger(__name__)

# ── Feature specification ─────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    COL_TONNAGE_IMPORT,
    COL_TONNAGE_EXPORT,
    COL_GROSS_TONNAGE,
    COL_LENGTH,
    COL_DRAFT,
    COL_TUGS_BERTHING,
    COL_ARRIVAL_MONTH,
    COL_ARRIVAL_DOW,
]

VESSEL_TYPE_PREFIX = "vtype"
CARGO_PREFIX       = "cargo"


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the numeric feature columns from the clean pipeline dataframe."""
    return df[NUMERIC_FEATURES].fillna(0)


def extract_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one-hot encoded vessel-type and cargo columns already in df."""
    type_cols  = [c for c in df.columns if c.startswith(VESSEL_TYPE_PREFIX + "_")]
    cargo_cols = [c for c in df.columns if c.startswith(CARGO_PREFIX + "_")]
    return df[type_cols + cargo_cols].fillna(0)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Combine numeric and one-hot features into the final ML input matrix."""
    numeric = extract_numeric_features(df)
    one_hot = extract_one_hot_features(df)
    return pd.concat([numeric, one_hot], axis=1)


def extract_target_vector(df: pd.DataFrame) -> pd.Series:
    """Return the service time column as the prediction target."""
    return df[COL_SERVICE_HOURS]


# ── Model building ────────────────────────────────────────────────────────────

def _build_linear_regression_pipeline() -> Pipeline:
    """Wrap LinearRegression in a scaling pipeline for numeric stability."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])


def _build_xgboost_pipeline() -> Pipeline:
    """XGBoost regressor pipeline — no scaling needed but kept consistent."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBRegressor(
            n_estimators      = 400,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            random_state      = RANDOM_STATE,
            verbosity         = 0,
        )),
    ])


def select_model_pipeline(use_production_model: bool) -> Pipeline:
    """Return the right pipeline based on environment and library availability."""
    if use_production_model and XGBOOST_AVAILABLE:
        logger.info("Using XGBoost production model")
        return _build_xgboost_pipeline()
    logger.info("Using LinearRegression baseline model")
    return _build_linear_regression_pipeline()


# ── Training ──────────────────────────────────────────────────────────────────

def split_train_test(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Time-safe split: earlier rows train, later rows test (no shuffle)."""
    split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, and R² for a set of predictions."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def train_service_time_model(df: pd.DataFrame,
                              use_production_model: bool = True) -> tuple[Pipeline, dict]:
    """
    Train the service time predictor on clean pipeline data.
    Returns (fitted_pipeline, evaluation_metrics).
    """
    X       = build_feature_matrix(df)
    y       = extract_target_vector(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = select_model_pipeline(use_production_model)
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    logger.info("Model trained — RMSE=%.2f h  MAE=%.2f h  R²=%.3f",
                metrics["rmse"], metrics["mae"], metrics["r2"])
    return pipeline, metrics


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, model_path: Path) -> None:
    """Serialize the fitted pipeline to disk with joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Model saved → %s", model_path)


def load_model(model_path: Path) -> Pipeline:
    """Deserialize the fitted pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run training first.")
    pipeline = joblib.load(model_path)
    logger.info("Model loaded ← %s", model_path)
    return pipeline


# ── Inference interface ───────────────────────────────────────────────────────

class ServiceTimePredictor:
    """
    Single-responsibility class: predict service time for one Vessel.
    Hides all feature engineering from callers.
    """

    def __init__(self, model_path: Path = SERVICE_TIME_MODEL) -> None:
        self._pipeline      = load_model(model_path)
        self._feature_names = self._pipeline.feature_names_in_ \
                              if hasattr(self._pipeline, "feature_names_in_") else None

    def predict_service_hours(self, vessel: Vessel) -> float:
        """Return predicted quay occupation time in hours for the given vessel."""
        feature_row = self._vessel_to_feature_row(vessel)
        prediction  = self._pipeline.predict(feature_row)[0]
        return float(max(prediction, 0.5))   # floor at 30 min

    def _vessel_to_feature_row(self, vessel: Vessel) -> pd.DataFrame:
        """Convert a Vessel object into the model's expected feature DataFrame."""
        base_features = {
            COL_TONNAGE_IMPORT: vessel.tonnage_import,
            COL_TONNAGE_EXPORT: vessel.tonnage_export,
            COL_GROSS_TONNAGE:  vessel.gross_tonnage,
            COL_LENGTH:         vessel.length_m,
            COL_DRAFT:          vessel.draft_m,
            COL_TUGS_BERTHING:  vessel.tugs_at_berthing,
            COL_ARRIVAL_MONTH:  vessel.arrival_month,
            COL_ARRIVAL_DOW:    vessel.arrival_dow,
        }
        row = pd.DataFrame([base_features])
        row = self._add_missing_one_hot_columns(row)
        return row

    def _add_missing_one_hot_columns(self, row: pd.DataFrame) -> pd.DataFrame:
        """Fill any one-hot columns the model expects but the row lacks."""
        if self._feature_names is None:
            return row
        for col in self._feature_names:
            if col not in row.columns:
                row[col] = 0
        return row[list(self._feature_names)]
