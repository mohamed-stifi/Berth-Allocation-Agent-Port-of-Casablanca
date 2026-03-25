"""
Layer 1 — Data Pipeline & Feature Engineering.

Responsibilities:
  1. Load raw CSV and JSON.
  2. Parse all datetime columns into proper objects.
  3. Derive KPI columns (waiting time, service time, anchorage flag).
  4. Encode categoricals for ML.
  5. Build the berth constraint lookup table from JSON.

Each function is ≤ 10 lines and has a single responsibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import (
    COL_ARRIVAL_DT, COL_ARRIVAL_DOW, COL_ARRIVAL_MONTH,
    COL_ANCHORAGE_FLAG, COL_BERTH_ID,
    COL_BERTHING_DT, COL_DATE_BERTHING, COL_DATE_PORT_ENTRY,
    COL_DATE_ANCHORAGE_IN,
    COL_DATE_UNBERTHING, COL_SERVICE_HOURS, COL_STAY_QUAY_DAYS,
    COL_TIME_BERTHING, COL_TIME_PORT_ENTRY, COL_TIME_UNBERTHING,
    COL_UNBERTHING_DT, COL_WAITING_HOURS,
    MAX_SERVICE_HOURS, MIN_SERVICE_HOURS,
)
from config.models import Berth

logger = logging.getLogger(__name__)


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Read the historical vessel call CSV from disk."""
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded CSV: %d rows, %d columns from %s", *df.shape, csv_path)
    return df


def load_port_json(json_path: Path) -> dict:
    """Read the port digital-twin JSON from disk."""
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded port JSON from %s", json_path)
    return data


# ── Datetime parsing ─────────────────────────────────────────────────────────

def _combine_date_and_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    """Merge a date column and a time column into a single datetime Series."""
    combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
    return pd.to_datetime(combined, dayfirst=True, errors="coerce")


def parse_event_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add arrival, berthing, and unberthing datetime columns to the frame."""
    df = df.copy()
    df[COL_ARRIVAL_DT]    = _combine_date_and_time(df, COL_DATE_PORT_ENTRY, COL_TIME_PORT_ENTRY)
    df[COL_BERTHING_DT]   = _combine_date_and_time(df, COL_DATE_BERTHING,   COL_TIME_BERTHING)
    df[COL_UNBERTHING_DT] = _combine_date_and_time(df, COL_DATE_UNBERTHING, COL_TIME_UNBERTHING)
    return df


# ── KPI derivation ────────────────────────────────────────────────────────────

def derive_waiting_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hours between port entry and actual berthing (= anchorage wait)."""
    df = df.copy()
    delta_seconds         = (df[COL_BERTHING_DT] - df[COL_ARRIVAL_DT]).dt.total_seconds()
    df[COL_WAITING_HOURS] = (delta_seconds / 3600).clip(lower=0)
    return df


def derive_service_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hours a vessel occupies the quay (= berthing to unberthing).

    Uses datetime difference as primary source. When datetime diff is zero or NaN
    (e.g. same-day departure without proper timestamp), falls back to
    Sejour Quai (Jours) × 24 hours.  Handles French decimal format (commas).
    """
    df = df.copy()
    delta_seconds = (df[COL_UNBERTHING_DT] - df[COL_BERTHING_DT]).dt.total_seconds()
    service_hours_dt = (delta_seconds / 3600).clip(lower=0)

    raw_days = df[COL_STAY_QUAY_DAYS].astype(str).str.replace(",", ".", regex=False)
    days_col = pd.to_numeric(raw_days, errors="coerce")
    service_hours_days = (days_col * 24).fillna(0)

    missing = (service_hours_dt == 0) | service_hours_dt.isna()
    service_hours = service_hours_dt.where(~missing, service_hours_days)
    service_hours = service_hours.clip(lower=0)
    df[COL_SERVICE_HOURS] = service_hours
    return df


def flag_anchorage_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Mark 1 if a vessel used the anchorage area before berthing, else 0."""
    df = df.copy()
    df[COL_ANCHORAGE_FLAG] = df[COL_DATE_ANCHORAGE_IN].notna().astype(int)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract month and day-of-week from arrival datetime for seasonality."""
    df = df.copy()
    df[COL_ARRIVAL_MONTH] = df[COL_ARRIVAL_DT].dt.month
    df[COL_ARRIVAL_DOW]   = df[COL_ARRIVAL_DT].dt.dayofweek
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def remove_service_time_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where service time is physically implausible."""
    mask = df[COL_SERVICE_HOURS].between(MIN_SERVICE_HOURS, MAX_SERVICE_HOURS)
    clean_df = df[mask].copy()
    n_dropped = len(df) - len(clean_df)
    logger.info("Removed %d outlier rows (service time out of [%.1f, %.1f] h)",
                n_dropped, MIN_SERVICE_HOURS, MAX_SERVICE_HOURS)
    return clean_df


def drop_rows_missing_critical_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that lack arrival or berthing datetime (cannot be used)."""
    required = [COL_ARRIVAL_DT, COL_BERTHING_DT, COL_UNBERTHING_DT]
    clean_df = df.dropna(subset=required).copy()
    logger.info("Dropped %d rows missing critical datetime fields", len(df) - len(clean_df))
    return clean_df


def normalize_berth_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from berth IDs so they match JSON keys exactly."""
    df = df.copy()
    df[COL_BERTH_ID] = df[COL_BERTH_ID].astype(str).str.strip()
    return df


# ── Categorical encoding ──────────────────────────────────────────────────────

def one_hot_encode_column(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Replace a categorical column with one-hot dummy columns."""
    dummies = pd.get_dummies(df[column].fillna("UNKNOWN"), prefix=prefix)
    return pd.concat([df.drop(columns=[column]), dummies], axis=1)


# ── Berth constraint matrix (from JSON) ───────────────────────────────────────

def _parse_single_berth(berth_id: str, berth_data: dict, terminal_name: str,
                         terminal_specs: dict, operator: str) -> Berth:
    """Construct a Berth domain object from its JSON representation."""
    allowed_cargos = frozenset(berth_data.get("cargo_types", []))
    allowed_types  = frozenset(berth_data.get("vessel_types", []))
    return Berth(
        berth_id      = berth_id,
        terminal_name = terminal_name,
        quay_length_m = float(terminal_specs.get("quay_length_m", 0)),
        depth_m       = float(str(terminal_specs.get("depth_m", 0)).split()[0]),
        allowed_cargos= allowed_cargos,
        allowed_types = allowed_types,
        operator      = operator or "UNKNOWN",
    )


RANGE_KEYS: dict[str, tuple[list[str], dict, str]] = {
    "30_to_36": (["30","31","32","33","34","35","36"],
                  {"quay_length_m": 1500, "depth_m": 9.0},
                  "Main General Cargo Terminal"),
    "40_to_44": (["40","41","42","43","44"],
                  {"quay_length_m": 400, "depth_m": 8.0},
                  "Ro-Ro & Passenger Terminal"),
    "A2_to_A5": (["A2","A3","A4","A5"],
                  {"quay_length_m": 300, "depth_m": 8.0},
                  "Ro-Ro & Passenger Terminal"),
    "50_to_54": (["50","51","52","53","54"],
                  {"quay_length_m": 700, "depth_m": 9.2},
                  "Container Terminal (SOMAPORT)"),
    "70_to_74": (["70","71","72","73","74"],
                  {"quay_length_m": 600, "depth_m": 12.0},
                  "East Container Terminal (Marsa Maroc)"),
    "80_to_83": (["80","81","82","83"],
                  {"quay_length_m": 600, "depth_m": 12.0},
                  "East Container Terminal (Marsa Maroc)"),
}

INDIVIDUAL_BERTH_SPECS: dict[str, dict] = {
    "20": {"quay_length_m": 300, "depth_m": 10.5},
    "21": {"quay_length_m": 300, "depth_m": 10.5},
    "60": {"quay_length_m": 300, "depth_m": 10.5},
    "61": {"quay_length_m": 300, "depth_m": 10.5},
}


def build_berth_constraint_matrix(port_json: dict) -> dict[str, Berth]:
    """Return a dict mapping berth_id → Berth for every berth in the JSON."""
    berth_matrix: dict[str, Berth] = {}
    port_data    = port_json["port_of_casablanca"]
    berths_info  = port_data.get("berths_details", {})
    terminals    = port_data.get("terminals", [])

    terminal_by_berth = _index_terminals_by_berth(terminals)

    for berth_id, berth_data in berths_info.items():
        if berth_id in RANGE_KEYS:
            expanded_ids, range_specs, range_name = RANGE_KEYS[berth_id]
            for exp_id in expanded_ids:
                term_info    = terminal_by_berth.get(exp_id, {})
                specs        = term_info.get("specifications", range_specs)
                name         = term_info.get("name", range_name)
                operator     = term_info.get("operator", "UNKNOWN")
                berth_obj    = _parse_single_berth(
                    berth_id      = exp_id,
                    berth_data    = berth_data,
                    terminal_name = name,
                    terminal_specs= specs,
                    operator      = operator,
                )
                berth_matrix[exp_id] = berth_obj
        else:
            terminal_info = terminal_by_berth.get(berth_id, {})
            term_specs    = terminal_info.get("specifications", {})
            fallback_specs = INDIVIDUAL_BERTH_SPECS.get(berth_id, {})
            specs = term_specs if term_specs else fallback_specs
            berth_obj = _parse_single_berth(
                berth_id      = berth_id,
                berth_data    = berth_data,
                terminal_name = terminal_info.get("name", "UNKNOWN"),
                terminal_specs= specs,
                operator      = terminal_info.get("operator", ""),
            )
            berth_matrix[berth_id] = berth_obj

    logger.info("Built berth constraint matrix: %d berths indexed", len(berth_matrix))
    return berth_matrix


def _index_terminals_by_berth(terminals: list[dict]) -> dict[str, dict]:
    """Build a reverse-index: berth_id → terminal dict."""
    index: dict[str, dict] = {}
    for terminal in terminals:
        for berth_id in terminal.get("berths", []):
            index[str(berth_id)] = terminal
    return index


# ── Master pipeline entry point ───────────────────────────────────────────────

def run_full_pipeline(csv_path: Path, json_path: Path) -> tuple[pd.DataFrame, dict[str, Berth]]:
    """
    Execute all pipeline steps in order.
    Returns (clean_feature_dataframe, berth_constraint_matrix).
    """
    raw_df       = load_raw_csv(csv_path)
    port_json    = load_port_json(json_path)
    berth_matrix = build_berth_constraint_matrix(port_json)

    df = (raw_df
          .pipe(parse_event_datetimes)
          .pipe(derive_waiting_time)
          .pipe(derive_service_time)
          .pipe(flag_anchorage_usage)
          .pipe(add_temporal_features)
          .pipe(drop_rows_missing_critical_fields)
          .pipe(remove_service_time_outliers)
          .pipe(normalize_berth_ids))

    logger.info("Pipeline complete: %d clean rows ready", len(df))
    return df, berth_matrix
