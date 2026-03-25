"""
Berth Allocation Agent — Command-Line Entry Point.

Usage:
  python main.py train                  # train ML model on historical CSV
  python main.py allocate               # run one allocation cycle (demo vessels)
  python main.py simulate               # run all 7 simulation scenarios
  python main.py dashboard              # launch Streamlit dashboard

All commands read paths from config/constants.py.
Set environment variable BERTH_ENV=production to use XGBoost instead of
LinearRegression baseline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Make sure all sub-packages are importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.constants import CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL
from config.models import Vessel

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

USE_PRODUCTION_MODEL = os.getenv("BERTH_ENV", "dev") == "production"


# ── Command handlers ──────────────────────────────────────────────────────────

def handle_train() -> None:
    """Offline training: process CSV, train predictor, save model to disk."""
    from agent import train_and_save_model
    logger.info("Starting training pipeline …")
    metrics = train_and_save_model(
        csv_path             = CSV_PATH,
        json_path            = JSON_PATH,
        model_path           = SERVICE_TIME_MODEL,
        use_production_model = USE_PRODUCTION_MODEL,
    )
    logger.info("Training complete.")
    logger.info("  RMSE = %.2f h", metrics["rmse"])
    logger.info("  MAE  = %.2f h", metrics["mae"])
    logger.info("  R²   = %.4f",   metrics["r2"])


def handle_allocate() -> None:
    """
    Online allocation demo: build agent, create a small synthetic queue,
    run the GA optimizer, and print the resulting schedule.
    """
    from agent import build_agent_from_config
    from pipeline.data_pipeline import run_full_pipeline

    logger.info("Building agent …")
    agent = build_agent_from_config(CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL)

    demo_queue = _build_demo_vessel_queue()
    logger.info("Allocating %d vessels …", len(demo_queue))
    schedule = agent.allocate(demo_queue)
    _print_schedule(schedule)


def handle_simulate() -> None:
    """Run all 7 validation scenarios and print the comparison table."""
    from agent import build_agent_from_config
    from pipeline.data_pipeline import run_full_pipeline
    from ml.service_time_predictor import ServiceTimePredictor

    logger.info("Loading data for simulation …")
    _, berth_matrix = run_full_pipeline(CSV_PATH, JSON_PATH)
    agent           = build_agent_from_config(CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL)

    demo_vessels    = _build_demo_vessel_queue()
    predictor       = ServiceTimePredictor(SERVICE_TIME_MODEL)
    service_times   = {v.name: predictor.predict_service_hours(v)
                       for v in demo_vessels}
    closure_start   = datetime(2024, 6, 5, 0, 0)

    results = agent.run_simulations(demo_vessels, service_times, closure_start)

    logger.info("\n%s", "─" * 80)
    logger.info("%-30s  %10s  %10s  %12s", "Scenario", "Avg wait", "Max wait", "Utilization")
    logger.info("─" * 80)
    for r in results:
        logger.info("%-30s  %9.1fh  %9.1fh  %11.1f%%",
                    r.scenario_name, r.avg_waiting_hours,
                    r.max_waiting_hours, r.berth_utilization * 100)


def handle_dashboard() -> None:
    """Launch the Streamlit dashboard (requires streamlit installed)."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "simulation_app.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)


# ── Demo data builder ─────────────────────────────────────────────────────────

def _build_demo_vessel_queue() -> list[Vessel]:
    """
    Construct a realistic synthetic queue of 8 vessels covering
    every terminal type at Casa port. Used for allocate + simulate demos.
    """
    base_eta = datetime(2024, 6, 1, 6, 0)

    return [
        Vessel(
            name="MSC_AURORA",        vessel_type="PORTE_CONTENEUR",
            gross_tonnage=75000,      length_m=294.0, draft_m=11.5,
            cargo_import="Containers", cargo_export="Containers",
            tonnage_import=12000.0,   tonnage_export=8000.0,
            tugs_at_berthing=3,       arrival_month=6, arrival_dow=5,
            eta=base_eta,
        ),
        Vessel(
            name="ATLAS_GRAIN",       vessel_type="VRAQUIER",
            gross_tonnage=28000,      length_m=185.0, draft_m=10.2,
            cargo_import="Grains",    cargo_export="",
            tonnage_import=22000.0,   tonnage_export=0.0,
            tugs_at_berthing=2,       arrival_month=6, arrival_dow=5,
            eta=base_eta + _hours(3),
        ),
        Vessel(
            name="OCP_PHOSPHATE_1",   vessel_type="VRAQUIER",
            gross_tonnage=45000,      length_m=220.0, draft_m=11.8,
            cargo_import="",          cargo_export="Phosphates",
            tonnage_import=0.0,       tonnage_export=35000.0,
            tugs_at_berthing=3,       arrival_month=6, arrival_dow=5,
            eta=base_eta + _hours(5),
        ),
        Vessel(
            name="MAERSK_CASABLANCA", vessel_type="PORTE_CONTENEUR",
            gross_tonnage=55000,      length_m=260.0, draft_m=12.0,
            cargo_import="Containers", cargo_export="Containers",
            tonnage_import=9000.0,    tonnage_export=6000.0,
            tugs_at_berthing=3,       arrival_month=6, arrival_dow=5,
            eta=base_eta + _hours(6),
        ),
        Vessel(
            name="MINERAL_STAR",      vessel_type="VRAQUIER",
            gross_tonnage=22000,      length_m=170.0, draft_m=9.5,
            cargo_import="Coal",      cargo_export="",
            tonnage_import=18000.0,   tonnage_export=0.0,
            tugs_at_berthing=2,       arrival_month=6, arrival_dow=5,
            eta=base_eta + _hours(8),
        ),
        Vessel(
            name="RORO_EXPRESS",      vessel_type="ROULIER",
            gross_tonnage=18000,      length_m=180.0, draft_m=7.5,
            cargo_import="Ro-Ro",     cargo_export="Ro-Ro",
            tonnage_import=3000.0,    tonnage_export=3000.0,
            tugs_at_berthing=1,       arrival_month=6, arrival_dow=5,
            eta=base_eta + _hours(10),
        ),
        Vessel(
            name="GENERAL_ATLAS",     vessel_type="CARGO_GENERAL",
            gross_tonnage=12000,      length_m=145.0, draft_m=8.0,
            cargo_import="General Cargo", cargo_export="",
            tonnage_import=7000.0,    tonnage_export=0.0,
            tugs_at_berthing=2,       arrival_month=6, arrival_dow=5,
            eta=base_eta + _hours(12),
        ),
        Vessel(
            name="CASTOR_CEREAL",     vessel_type="VRAQUIER",
            gross_tonnage=30000,      length_m=190.0, draft_m=10.0,
            cargo_import="Grains",    cargo_export="",
            tonnage_import=25000.0,   tonnage_export=0.0,
            tugs_at_berthing=2,       arrival_month=6, arrival_dow=5,
            eta=base_eta + _hours(14),
        ),
    ]


def _hours(n: int):
    from datetime import timedelta
    return timedelta(hours=n)


# ── Schedule printer ──────────────────────────────────────────────────────────

def _print_schedule(schedule) -> None:
    """Pretty-print the optimizer's Schedule to stdout."""
    print("\n" + "═" * 90)
    print(f"  RECOMMENDED SCHEDULE  |  Total wait: {schedule.total_wait_hours:.1f} h"
          f"  |  Avg wait: {schedule.avg_wait_hours:.1f} h")
    print("═" * 90)
    header = f"{'Vessel':<22}{'Berth':<8}{'Terminal':<22}{'ETB':<18}{'ETD':<18}{'Wait(h)':>7}"
    print(header)
    print("─" * 90)
    for a in sorted(schedule.assignments, key=lambda x: x.scheduled_start):
        print(
            f"{a.vessel.name:<22}"
            f"{a.berth.berth_id:<8}"
            f"{a.berth.terminal_name:<22}"
            f"{a.scheduled_start.strftime('%Y-%m-%d %H:%M'):<18}"
            f"{a.predicted_end.strftime('%Y-%m-%d %H:%M'):<18}"
            f"{a.predicted_wait_hours:>7.1f}"
        )
    print("═" * 90 + "\n")


# ── CLI parser ────────────────────────────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    """Define and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog        = "berth_agent",
        description = "Port of Casablanca — Berth Allocation Agent",
    )
    parser.add_argument(
        "command",
        choices = ["train", "allocate", "simulate", "dashboard"],
        help    = "train | allocate | simulate | dashboard",
    )
    return parser


def dispatch_command(command: str) -> None:
    """Route a CLI command string to the correct handler function."""
    handlers = {
        "train":     handle_train,
        "allocate":  handle_allocate,
        "simulate":  handle_simulate,
        "dashboard": handle_dashboard,
    }
    handlers[command]()


def main() -> None:
    parser  = build_argument_parser()
    args    = parser.parse_args()
    dispatch_command(args.command)


if __name__ == "__main__":
    main()
