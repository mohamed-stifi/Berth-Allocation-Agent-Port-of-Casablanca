"""
KPI Reporter — computes, aggregates, and formats all performance metrics
for the Berth Allocation Agent.

This module is purely analytical: it reads Schedules and SimulationResults
and produces structured reports. It never modifies any state.

KPIs tracked:
  1. Average waiting time (hours)
  2. Maximum waiting time (hours)
  3. Berth utilization rate (%)
  4. Direct berthing rate — vessels with zero anchorage wait (%)
  5. Schedule accuracy — vessels berthed within 1 h of predicted ETB (%)
  6. ML RMSE / MAE / R² (from training metrics)
  7. Queue length evolution over time
  8. Vessels served per 24-hour period
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from config.models import Assignment, Schedule
from simulation.scenario_engine import SimulationResult

logger = logging.getLogger(__name__)

ON_TIME_WINDOW_HOURS       = 1.0    # vessel is "on time" if berths within 1 h of ETB
DIRECT_BERTHING_THRESHOLD  = 0.1    # hours — less than 6 min wait = direct berth
UTILIZATION_HORIZON_HOURS  = 24.0   # reference window for utilization computation


# ── Per-schedule KPIs ─────────────────────────────────────────────────────────

@dataclass
class ScheduleKPIs:
    """Aggregated KPIs for one optimizer Schedule."""
    total_vessels:          int
    avg_wait_hours:         float
    max_wait_hours:         float
    min_wait_hours:         float
    direct_berth_count:     int
    direct_berth_rate:      float    # fraction in [0, 1]
    total_service_hours:    float
    computed_at:            datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "total_vessels":       self.total_vessels,
            "avg_wait_hours":      round(self.avg_wait_hours,     2),
            "max_wait_hours":      round(self.max_wait_hours,     2),
            "min_wait_hours":      round(self.min_wait_hours,     2),
            "direct_berth_count":  self.direct_berth_count,
            "direct_berth_rate_%": round(self.direct_berth_rate * 100, 1),
            "total_service_hours": round(self.total_service_hours, 1),
        }


def compute_schedule_kpis(schedule: Schedule) -> ScheduleKPIs:
    """Extract and compute all KPIs from a single optimizer Schedule."""
    assignments   = schedule.assignments
    if not assignments:
        return ScheduleKPIs(0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    wait_times    = [a.predicted_wait_hours for a in assignments]
    service_hours = [
        (a.predicted_end - a.scheduled_start).total_seconds() / 3600
        for a in assignments
    ]
    direct_count  = sum(1 for w in wait_times if w < DIRECT_BERTHING_THRESHOLD)

    return ScheduleKPIs(
        total_vessels       = len(assignments),
        avg_wait_hours      = sum(wait_times) / len(wait_times),
        max_wait_hours      = max(wait_times),
        min_wait_hours      = min(wait_times),
        direct_berth_count  = direct_count,
        direct_berth_rate   = direct_count / len(assignments),
        total_service_hours = sum(service_hours),
    )


# ── Schedule accuracy (requires ground truth) ─────────────────────────────────

def compute_schedule_accuracy(assignments: list[Assignment],
                               actual_berth_times: dict[str, datetime]) -> float:
    """
    Compare predicted ETB against actual berthing time for each vessel.
    Returns the fraction of vessels that berthed within ON_TIME_WINDOW_HOURS
    of the predicted ETB.

    actual_berth_times: vessel_name → actual datetime of berthing
    """
    if not assignments or not actual_berth_times:
        return 0.0

    on_time_count = 0
    for assignment in assignments:
        actual = actual_berth_times.get(assignment.vessel.name)
        if actual is None:
            continue
        error_hours = abs((assignment.scheduled_start - actual).total_seconds()) / 3600
        if error_hours <= ON_TIME_WINDOW_HOURS:
            on_time_count += 1

    matched = sum(1 for a in assignments if a.vessel.name in actual_berth_times)
    return on_time_count / matched if matched > 0 else 0.0


# ── Berth utilization over time ───────────────────────────────────────────────

def compute_berth_utilization(assignments: list[Assignment],
                               total_berths: int,
                               horizon_hours: float = UTILIZATION_HORIZON_HOURS) -> float:
    """
    Compute what fraction of berth-hours were actually occupied.
    utilization = sum(service_hours) / (total_berths × horizon_hours)
    """
    if not assignments or total_berths == 0:
        return 0.0

    total_service = sum(
        (a.predicted_end - a.scheduled_start).total_seconds() / 3600
        for a in assignments
    )
    return min(1.0, total_service / (total_berths * horizon_hours))


# ── Queue length time series ──────────────────────────────────────────────────

def build_queue_length_timeseries(assignments: list[Assignment],
                                   interval_hours: float = 1.0) -> pd.DataFrame:
    """
    Produce a time-series DataFrame showing queue depth at each hour.
    Useful for plotting congestion peaks on the dashboard.
    """
    if not assignments:
        return pd.DataFrame(columns=["timestamp", "queue_length"])

    start_time = min(a.vessel.eta         for a in assignments)
    end_time   = max(a.predicted_end      for a in assignments)
    timestamps = []
    queue_lens = []

    cursor = start_time
    while cursor <= end_time:
        queue_at_cursor = sum(
            1 for a in assignments
            if a.vessel.eta <= cursor < a.scheduled_start
        )
        timestamps.append(cursor)
        queue_lens.append(queue_at_cursor)
        cursor += timedelta(hours=interval_hours)

    return pd.DataFrame({"timestamp": timestamps, "queue_length": queue_lens})


# ── Scenario comparison table ─────────────────────────────────────────────────

def build_scenario_comparison_table(results: list[SimulationResult]) -> pd.DataFrame:
    """
    Turn a list of SimulationResults into a DataFrame suitable for
    display in the dashboard or saving to CSV.
    """
    rows = []
    for r in results:
        rows.append({
            "Scenario":              r.scenario_name,
            "Vessels served":        r.total_vessels_served,
            "Avg wait (h)":          round(r.avg_waiting_hours,      2),
            "Max wait (h)":          round(r.max_waiting_hours,      2),
            "Utilization (%)":       round(r.berth_utilization * 100, 1),
            "Direct berths":         r.vessels_with_zero_wait,
            "Direct berth rate (%)": round(
                r.vessels_with_zero_wait / r.total_vessels_served * 100
                if r.total_vessels_served > 0 else 0, 1),
        })
    return pd.DataFrame(rows).set_index("Scenario")


# ── ML metrics formatter ──────────────────────────────────────────────────────

def format_ml_metrics(metrics: dict) -> str:
    """Format ML evaluation metrics into a human-readable string."""
    return (
        f"Service Time Predictor — "
        f"RMSE={metrics['rmse']:.2f}h  "
        f"MAE={metrics['mae']:.2f}h  "
        f"R²={metrics['r2']:.4f}"
    )


# ── Full report ───────────────────────────────────────────────────────────────

def print_full_kpi_report(schedule: Schedule,
                           ml_metrics: dict | None = None,
                           scenario_results: list[SimulationResult] | None = None) -> None:
    """Print a formatted KPI report to the logger."""
    kpis = compute_schedule_kpis(schedule)

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║      BERTH ALLOCATION KPI REPORT     ║")
    logger.info("╠══════════════════════════════════════╣")
    logger.info("║  Vessels scheduled : %d",               kpis.total_vessels)
    logger.info("║  Avg wait time     : %.2f h",            kpis.avg_wait_hours)
    logger.info("║  Max wait time     : %.2f h",            kpis.max_wait_hours)
    logger.info("║  Direct berthing   : %d (%.1f%%)",
                kpis.direct_berth_count, kpis.direct_berth_rate * 100)
    logger.info("║  Total service hrs : %.1f h",            kpis.total_service_hours)
    if ml_metrics:
        logger.info("║  %s", format_ml_metrics(ml_metrics))
    logger.info("╚══════════════════════════════════════╝")

    if scenario_results:
        df = build_scenario_comparison_table(scenario_results)
        logger.info("\nScenario comparison:\n%s", df.to_string())
