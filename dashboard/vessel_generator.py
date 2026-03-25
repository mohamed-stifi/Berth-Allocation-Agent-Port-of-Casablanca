"""
dashboard/vessel_generator.py
─────────────────────────────
Generates realistic random Vessel objects for simulation seeding.

Rules:
  - Each generated vessel is physically consistent (draft ≤ port max, etc.)
  - Cargo type and vessel type are always compatible pairs
  - Names follow a realistic pattern (shipowner prefix + random suffix)
  - No magic values — all ranges come from config.constants
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta

from config.constants import PORT_MAX_DRAFT_M, PORT_MIN_DRAFT_M
from config.models import Vessel

# ── Compatible (vessel_type, cargo_import, cargo_export) triplets ─────────────

VESSEL_PROFILES = [
    {
        "vessel_type":    "PORTE_CONTENEUR",
        "cargo_import":   "Containers",
        "cargo_export":   "Containers",
        "length_range":   (150.0, 300.0),
        "draft_range":    (9.5, 13.0),
        "tonnage_range":  (15000, 80000),
        "tugs_range":     (2, 4),
        "prefixes":       ["MSC", "MAERSK", "CMA", "EVERGREEN", "HAPAG", "COSCO"],
    },
    {
        "vessel_type":    "VRAQUIER",
        "cargo_import":   "Grains",
        "cargo_export":   "",
        "length_range":   (150.0, 220.0),
        "draft_range":    (8.5, 12.0),
        "tonnage_range":  (20000, 60000),
        "tugs_range":     (2, 3),
        "prefixes":       ["ATLAS", "CASTOR", "ORION", "DELTA", "SIGMA"],
    },
    {
        "vessel_type":    "VRAQUIER",
        "cargo_import":   "Coal",
        "cargo_export":   "",
        "length_range":   (140.0, 200.0),
        "draft_range":    (8.0, 11.0),
        "tonnage_range":  (15000, 45000),
        "tugs_range":     (2, 3),
        "prefixes":       ["MINERAL", "BULK", "NORDIC", "PACIFIC"],
    },
    {
        "vessel_type":    "VRAQUIER",
        "cargo_import":   "",
        "cargo_export":   "Phosphates",
        "length_range":   (170.0, 240.0),
        "draft_range":    (9.0, 12.5),
        "tonnage_range":  (25000, 55000),
        "tugs_range":     (2, 4),
        "prefixes":       ["OCP", "PHOSPHO", "ATLAS"],
    },
    {
        "vessel_type":    "CARGO_GENERAL",
        "cargo_import":   "General Cargo",
        "cargo_export":   "",
        "length_range":   (100.0, 180.0),
        "draft_range":    (7.0, 10.0),
        "tonnage_range":  (5000, 25000),
        "tugs_range":     (1, 2),
        "prefixes":       ["GENERAL", "CARGO", "ATLANTIC", "AFRICA"],
    },
    {
        "vessel_type":    "ROULIER",
        "cargo_import":   "Ro-Ro",
        "cargo_export":   "Ro-Ro",
        "length_range":   (140.0, 200.0),
        "draft_range":    (6.5, 9.0),
        "tonnage_range":  (10000, 30000),
        "tugs_range":     (1, 2),
        "prefixes":       ["RORO", "TRAILER", "AUTO", "GRIMALDI"],
    },
]

VESSEL_TYPE_LABELS = {
    "PORTE_CONTENEUR": "Container Ship",
    "VRAQUIER":        "Bulk Carrier",
    "CARGO_GENERAL":   "General Cargo",
    "ROULIER":         "Ro-Ro",
}

_used_names: set[str] = set()


def _unique_vessel_name(prefix: str) -> str:
    """Generate a unique vessel name like 'MSC_AURORA_7K2'."""
    for _ in range(100):
        suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
        name   = f"{prefix}_{suffix}"
        if name not in _used_names:
            _used_names.add(name)
            return name
    return f"{prefix}_{random.randint(1000, 9999)}"


def _rand_float(lo: float, hi: float, decimals: int = 1) -> float:
    return round(random.uniform(lo, hi), decimals)


def generate_random_vessel(eta: datetime | None = None,
                            profile_index: int | None = None) -> Vessel:
    """
    Return one randomly generated Vessel with realistic, internally
    consistent physical and operational attributes.
    """
    profile = (VESSEL_PROFILES[profile_index]
               if profile_index is not None
               else random.choice(VESSEL_PROFILES))

    prefix   = random.choice(profile["prefixes"])
    name     = _unique_vessel_name(prefix)
    length   = _rand_float(*profile["length_range"])
    draft    = min(_rand_float(*profile["draft_range"]), PORT_MAX_DRAFT_M)
    draft    = max(draft, PORT_MIN_DRAFT_M)
    tonnage  = random.randint(*profile["tonnage_range"])
    tugs     = random.randint(*profile["tugs_range"])
    imp_ton  = round(tonnage * random.uniform(0.3, 0.9))
    exp_ton  = round(tonnage * random.uniform(0.0, 0.4))
    eta_dt   = eta or datetime.utcnow() + timedelta(hours=random.uniform(0, 12))

    return Vessel(
        name             = name,
        vessel_type      = profile["vessel_type"],
        gross_tonnage    = tonnage,
        length_m         = length,
        draft_m          = draft,
        cargo_import     = profile["cargo_import"],
        cargo_export     = profile["cargo_export"],
        tonnage_import   = float(imp_ton),
        tonnage_export   = float(exp_ton),
        tugs_at_berthing = tugs,
        arrival_month    = eta_dt.month,
        arrival_dow      = eta_dt.weekday(),
        eta              = eta_dt,
    )


def generate_vessel_fleet(count: int,
                           base_time: datetime,
                           spread_hours: float = 24.0) -> list[Vessel]:
    """
    Generate `count` vessels with ETAs spread across `spread_hours` from base_time.
    Used to seed the waiting queue at simulation start.
    """
    vessels = []
    for i in range(count):
        offset = random.uniform(0, spread_hours)
        eta    = base_time + timedelta(hours=offset)
        vessels.append(generate_random_vessel(eta=eta))
    return sorted(vessels, key=lambda v: v.eta)


def vessel_type_options() -> list[str]:
    """Return the list of all supported vessel type codes."""
    return list({p["vessel_type"] for p in VESSEL_PROFILES})


def cargo_options_for_type(vessel_type: str) -> tuple[list[str], list[str]]:
    """Return (import_options, export_options) for a given vessel type."""
    matches = [p for p in VESSEL_PROFILES if p["vessel_type"] == vessel_type]
    imports = list({p["cargo_import"] for p in matches} - {""})
    exports = list({p["cargo_export"] for p in matches} - {""})
    return imports or ["General Cargo"], exports or [""]
