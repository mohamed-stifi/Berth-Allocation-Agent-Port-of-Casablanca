"""
Berth Allocation Agent — Orchestrator.

This is the single integration point that wires every module together.
It follows the Facade pattern: callers interact only with BerthAllocationAgent
and never directly with ML, optimizer, tracker, or constraint engine.

Two public workflows:
  1. train()   — offline training on historical CSV data
  2. allocate() — online: predict service times, filter constraints,
                  run GA, write to tracker, return Schedule

SOLID:
  - Single responsibility : orchestration only, zero business logic
  - Open/Closed           : new modules plug in through __init__ injection
  - Dependency inversion  : depends on abstractions (domain models), not
                            concrete ML or GA internals
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from config.constants import CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL
from config.models import Berth, Schedule, Vessel
from constraints.constraint_engine import ConstraintEngine, EligibilityReport
from ml.service_time_predictor import (
    ServiceTimePredictor,
    build_feature_matrix,
    extract_target_vector,
    save_model,
    train_service_time_model,
)
from optimizer.genetic_algorithm import (
    chromosome_to_schedule,
    run_genetic_algorithm,
)
from pipeline.data_pipeline import run_full_pipeline
from simulation.scenario_engine import run_all_scenarios, SimulationResult
from tracker.port_tracker import PortTracker

logger = logging.getLogger(__name__)


class BerthAllocationAgent:
    """
    Top-level agent: orchestrates data pipeline → ML → constraints → GA → tracker.
    Inject dependencies through the constructor for testability.
    """

    def __init__(self,
                 berth_matrix:      dict[str, Berth],
                 port_tracker:      PortTracker,
                 constraint_engine: ConstraintEngine,
                 predictor:         ServiceTimePredictor) -> None:
        self._berth_matrix       = berth_matrix
        self._tracker            = port_tracker
        self._constraint_engine  = constraint_engine
        self._predictor          = predictor

    # ── Primary online workflow ───────────────────────────────────────────────

    def allocate(self, vessel_queue: list[Vessel]) -> Schedule:
        """
        Given a list of vessels waiting for a berth, return the
        optimal schedule that minimises total anchorage waiting time.
        """
        logger.info("Starting allocation for %d vessels", len(vessel_queue))

        service_times   = self._predict_all_service_times(vessel_queue)
        eligible_berths = self._filter_eligible_berths(vessel_queue)

        feasible_vessels = self._drop_infeasible_vessels(
            vessel_queue, eligible_berths)

        best_chromosome, _ = run_genetic_algorithm(
            vessels         = feasible_vessels,
            eligible_berths = eligible_berths,
            service_times   = service_times,
            tracker         = self._tracker,
        )

        schedule = chromosome_to_schedule(
            chromosome    = best_chromosome,
            vessels       = feasible_vessels,
            berth_matrix  = self._berth_matrix,
            service_times = service_times,
        )

        self._register_schedule_with_tracker(schedule)
        logger.info("Allocation complete: %d assignments, avg wait = %.1f h",
                    len(schedule.assignments), schedule.avg_wait_hours)
        return schedule

    # ── Sub-steps ─────────────────────────────────────────────────────────────

    def _predict_all_service_times(self,
                                    vessels: list[Vessel]) -> dict[str, float]:
        """Call the ML predictor for every vessel in the queue."""
        return {
            vessel.name: self._predictor.predict_service_hours(vessel)
            for vessel in vessels
        }

    def _filter_eligible_berths(self,
                                 vessels: list[Vessel]
                                 ) -> dict[str, list[Berth]]:
        """Run the constraint engine for every vessel."""
        return {
            vessel.name: self._constraint_engine.find_eligible_berths(vessel).eligible_berths
            for vessel in vessels
        }

    def _drop_infeasible_vessels(self,
                                  vessels: list[Vessel],
                                  eligible_berths: dict[str, list[Berth]]
                                  ) -> list[Vessel]:
        """Remove vessels with no eligible berths — cannot be scheduled."""
        feasible = [v for v in vessels if eligible_berths.get(v.name)]
        skipped  = len(vessels) - len(feasible)
        if skipped:
            logger.warning("Dropped %d vessels with no eligible berths", skipped)
        return feasible

    def _register_schedule_with_tracker(self, schedule: Schedule) -> None:
        """Write every assignment into the Port Tracker's berth timelines."""
        for assignment in schedule.assignments:
            self._tracker.assign_vessel_to_berth(
                vessel     = assignment.vessel,
                berth_id   = assignment.berth.berth_id,
                start_time = assignment.scheduled_start,
                end_time   = assignment.predicted_end,
            )

    # ── Simulation workflow ───────────────────────────────────────────────────

    def run_simulations(self,
                        historical_vessels: list[Vessel],
                        service_times:      dict[str, float],
                        closure_start:      datetime) -> list[SimulationResult]:
        """Execute all 7 validation scenarios and return their results."""
        total_berths = len(self._berth_matrix)
        return run_all_scenarios(historical_vessels, service_times,
                                  total_berths, closure_start, self._berth_matrix)

    # ── Snapshot for dashboard ────────────────────────────────────────────────

    def get_port_snapshot(self, at_time: datetime,
                          vessel_queue: list[Vessel]):
        """Delegate to Port Tracker for a dashboard-ready snapshot."""
        return self._tracker.get_port_snapshot(at_time, vessel_queue)


# ── Factory / dependency injection ───────────────────────────────────────────

def build_agent_from_config(csv_path:    Path = CSV_PATH,
                             json_path:   Path = JSON_PATH,
                             model_path:  Path = SERVICE_TIME_MODEL) -> BerthAllocationAgent:
    """
    Construct a fully wired BerthAllocationAgent from file paths.
    Use this factory in production; inject mocks in tests.
    """
    from pipeline.data_pipeline import run_full_pipeline
    _, berth_matrix = run_full_pipeline(csv_path, json_path)

    tracker           = PortTracker(berth_matrix)
    constraint_engine = ConstraintEngine(berth_matrix)
    predictor         = ServiceTimePredictor(model_path)

    return BerthAllocationAgent(
        berth_matrix      = berth_matrix,
        port_tracker      = tracker,
        constraint_engine = constraint_engine,
        predictor         = predictor,
    )


# ── Offline training entry point ──────────────────────────────────────────────

def train_and_save_model(csv_path:   Path = CSV_PATH,
                          json_path:  Path = JSON_PATH,
                          model_path: Path = SERVICE_TIME_MODEL,
                          use_production_model: bool = True) -> dict:
    """
    Run the full training pipeline and persist the model.
    Returns evaluation metrics dict.
    Call this once before running the agent live.
    """
    import pandas as pd
    from pipeline.data_pipeline import run_full_pipeline
    from pipeline.data_pipeline import one_hot_encode_column
    from config.constants import COL_VESSEL_TYPE, COL_CARGO_IMPORT
    from ml.service_time_predictor import VESSEL_TYPE_PREFIX, CARGO_PREFIX

    df, _ = run_full_pipeline(csv_path, json_path)
    df    = one_hot_encode_column(df, COL_VESSEL_TYPE,   VESSEL_TYPE_PREFIX)
    df    = one_hot_encode_column(df, COL_CARGO_IMPORT,  CARGO_PREFIX)

    pipeline, metrics = train_service_time_model(df, use_production_model)
    save_model(pipeline, model_path)
    return metrics
