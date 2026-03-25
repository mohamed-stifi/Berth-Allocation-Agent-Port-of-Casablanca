"""
Vessel Builder — converts a raw pandas DataFrame row (or a dict from
a live AIS/VTS feed) into a typed Vessel domain object.

This is the bridge between the data pipeline and the agent's runtime.
Keeping it separate obeys Single Responsibility: the pipeline cleans
data, this module shapes it into domain objects.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from config.constants import (
    COL_ARRIVAL_DT, COL_ARRIVAL_DOW, COL_ARRIVAL_MONTH,
    COL_CARGO_EXPORT, COL_CARGO_IMPORT,
    COL_DRAFT, COL_GROSS_TONNAGE, COL_LENGTH,
    COL_TONNAGE_EXPORT, COL_TONNAGE_IMPORT,
    COL_TUGS_BERTHING, COL_VESSEL_NAME, COL_VESSEL_TYPE,
)
from config.models import Vessel

logger = logging.getLogger(__name__)

FALLBACK_DRAFT_M   = 8.0
FALLBACK_LENGTH_M  = 150.0
FALLBACK_TONNAGE   = 0.0
FALLBACK_TUGS      = 1
FALLBACK_TYPE      = "CARGO_GENERAL"
FALLBACK_CARGO     = "General Cargo"


# ── Single row → Vessel ───────────────────────────────────────────────────────

def row_to_vessel(row: pd.Series) -> Vessel:
    """
    Convert one cleaned DataFrame row into a Vessel domain object.
    Uses fallback values for any null field so the agent never crashes
    on incomplete records.
    """
    return Vessel(
        name             = str(row[COL_VESSEL_NAME]),
        vessel_type      = _safe_str(row, COL_VESSEL_TYPE,  FALLBACK_TYPE),
        gross_tonnage    = int(_safe_num(row, COL_GROSS_TONNAGE, 0)),
        length_m         = float(_safe_num(row, COL_LENGTH,       FALLBACK_LENGTH_M)),
        draft_m          = float(_safe_num(row, COL_DRAFT,        FALLBACK_DRAFT_M)),
        cargo_import     = _safe_str(row, COL_CARGO_IMPORT,  FALLBACK_CARGO),
        cargo_export     = _safe_str(row, COL_CARGO_EXPORT,  ""),
        tonnage_import   = float(_safe_num(row, COL_TONNAGE_IMPORT, FALLBACK_TONNAGE)),
        tonnage_export   = float(_safe_num(row, COL_TONNAGE_EXPORT, FALLBACK_TONNAGE)),
        tugs_at_berthing = int(_safe_num(row, COL_TUGS_BERTHING,    FALLBACK_TUGS)),
        arrival_month    = int(_safe_num(row, COL_ARRIVAL_MONTH, datetime.utcnow().month)),
        arrival_dow      = int(_safe_num(row, COL_ARRIVAL_DOW,   datetime.utcnow().weekday())),
        eta              = _safe_datetime(row, COL_ARRIVAL_DT),
    )


# ── DataFrame → list[Vessel] ──────────────────────────────────────────────────

def dataframe_to_vessel_queue(df: pd.DataFrame,
                               max_vessels: int | None = None) -> list[Vessel]:
    """
    Convert a clean pipeline DataFrame into an ordered list of Vessels.
    Sorted by ETA ascending so the optimizer receives a chronological queue.
    Optionally cap at max_vessels for demo / testing.
    """
    sorted_df = df.sort_values(COL_ARRIVAL_DT).reset_index(drop=True)
    if max_vessels is not None:
        sorted_df = sorted_df.head(max_vessels)

    vessels   = []
    failures  = 0
    for _, row in sorted_df.iterrows():
        try:
            vessels.append(row_to_vessel(row))
        except Exception as exc:
            failures += 1
            logger.warning("Skipped row for vessel '%s': %s",
                           row.get(COL_VESSEL_NAME, "UNKNOWN"), exc)

    logger.info("Built vessel queue: %d vessels (%d skipped)", len(vessels), failures)
    return vessels


# ── AIS live feed → Vessel ────────────────────────────────────────────────────

def ais_message_to_vessel(ais_payload: dict) -> Vessel:
    """
    Convert a live AIS/VTS message dict into a Vessel domain object.

    Expected keys (all optional except 'name' and 'eta'):
      name, vessel_type, gross_tonnage, length_m, draft_m,
      cargo_import, cargo_export, tonnage_import, tonnage_export,
      tugs_at_berthing, eta (ISO-8601 string or datetime)
    """
    eta_raw = ais_payload.get("eta", datetime.utcnow().isoformat())
    eta     = datetime.fromisoformat(str(eta_raw)) if isinstance(eta_raw, str) else eta_raw

    return Vessel(
        name             = ais_payload["name"],
        vessel_type      = ais_payload.get("vessel_type",      FALLBACK_TYPE),
        gross_tonnage    = int(ais_payload.get("gross_tonnage", 0)),
        length_m         = float(ais_payload.get("length_m",   FALLBACK_LENGTH_M)),
        draft_m          = float(ais_payload.get("draft_m",    FALLBACK_DRAFT_M)),
        cargo_import     = ais_payload.get("cargo_import",     FALLBACK_CARGO),
        cargo_export     = ais_payload.get("cargo_export",     ""),
        tonnage_import   = float(ais_payload.get("tonnage_import", FALLBACK_TONNAGE)),
        tonnage_export   = float(ais_payload.get("tonnage_export", FALLBACK_TONNAGE)),
        tugs_at_berthing = int(ais_payload.get("tugs_at_berthing", FALLBACK_TUGS)),
        arrival_month    = eta.month,
        arrival_dow      = eta.weekday(),
        eta              = eta,
    )


# ── Safe field accessors ──────────────────────────────────────────────────────

def _safe_str(row: pd.Series, col: str, fallback: str) -> str:
    """Return string value or fallback if null/missing."""
    val = row.get(col)
    return fallback if pd.isna(val) else str(val).strip()


def _safe_num(row: pd.Series, col: str, fallback: float) -> float:
    """Return numeric value or fallback if null/missing/non-numeric."""
    val = row.get(col)
    if pd.isna(val):
        return fallback
    try:
        return float(val)
    except (ValueError, TypeError):
        return fallback


def _safe_datetime(row: pd.Series, col: str) -> datetime:
    """Return datetime value or utcnow() if null."""
    val = row.get(col)
    if pd.isna(val) or val is None:
        logger.warning("Missing arrival datetime — defaulting to utcnow()")
        return datetime.utcnow()
    return val if isinstance(val, datetime) else pd.Timestamp(val).to_pydatetime()
