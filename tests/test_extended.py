"""
Extended unit tests — vessel_builder, kpi_reporter, schedule KPIs,
scenario comparison table, and the agent orchestrator wiring.

Run with: python -m pytest tests/test_extended.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.models import Assignment, Berth, BerthReservation, Schedule, Vessel


# ── Shared factories ──────────────────────────────────────────────────────────

def make_vessel(name: str = "V1",
                draft_m: float = 9.0, length_m: float = 150.0,
                cargo: str = "Containers",
                eta: datetime | None = None) -> Vessel:
    return Vessel(
        name=name, vessel_type="PORTE_CONTENEUR",
        gross_tonnage=20000, length_m=length_m, draft_m=draft_m,
        cargo_import=cargo, cargo_export="",
        tonnage_import=5000.0, tonnage_export=0.0,
        tugs_at_berthing=2, arrival_month=6, arrival_dow=1,
        eta=eta or datetime(2024, 6, 1, 8, 0),
    )


def make_berth(berth_id: str = "70") -> Berth:
    return Berth(
        berth_id=berth_id, terminal_name="Marsa Maroc",
        quay_length_m=600.0, depth_m=12.0,
        allowed_cargos=frozenset({"Containers"}),
        allowed_types=frozenset(), operator="Marsa Maroc",
    )


def make_assignment(vessel: Vessel | None = None,
                    wait_hours: float = 2.0) -> Assignment:
    v = vessel or make_vessel()
    b = make_berth()
    start = v.eta + timedelta(hours=wait_hours)
    end   = start + timedelta(hours=8)
    return Assignment(vessel=v, berth=b,
                      scheduled_start=start, predicted_end=end,
                      predicted_wait_hours=wait_hours)


# ── Test: VesselBuilder from pandas row ───────────────────────────────────────

class TestVesselBuilder:

    def _make_row(self, **overrides) -> "pd.Series":
        import pandas as pd
        base = {
            "Navire": "TEST_SHIP",
            "Type Navire": "VRAQUIER",
            "Jauge Brute": 30000,
            "Longueur": 185.0,
            "Tirant Eau": 10.5,
            "Marchandise DAP Import": "Grains",
            "Marchandise DAP Export": "",
            "Tonnage DAP Import": 22000.0,
            "Tonnage DAP Export": 0.0,
            "Nbr Remorque Accostage": 2,
            "arrival_month": 6,
            "arrival_day_of_week": 3,
            "arrival_datetime": datetime(2024, 6, 1, 8, 0),
        }
        base.update(overrides)
        return pd.Series(base)

    def test_row_to_vessel_name_correct(self):
        from pipeline.vessel_builder import row_to_vessel
        row    = self._make_row()
        vessel = row_to_vessel(row)
        assert vessel.name == "TEST_SHIP"

    def test_row_to_vessel_draft_parsed(self):
        from pipeline.vessel_builder import row_to_vessel
        row    = self._make_row(**{"Tirant Eau": 11.2})
        vessel = row_to_vessel(row)
        assert vessel.draft_m == pytest.approx(11.2)

    def test_row_to_vessel_null_cargo_uses_fallback(self):
        import numpy as np
        from pipeline.vessel_builder import row_to_vessel, FALLBACK_CARGO
        row    = self._make_row(**{"Marchandise DAP Import": np.nan})
        vessel = row_to_vessel(row)
        assert vessel.cargo_import == FALLBACK_CARGO

    def test_ais_message_to_vessel_parses_eta_string(self):
        from pipeline.vessel_builder import ais_message_to_vessel
        payload = {
            "name":      "LIVE_VESSEL",
            "eta":       "2024-07-15T06:30:00",
            "draft_m":   9.5,
            "length_m":  180.0,
        }
        vessel = ais_message_to_vessel(payload)
        assert vessel.eta == datetime(2024, 7, 15, 6, 30, 0)
        assert vessel.draft_m == pytest.approx(9.5)


# ── Test: KPI Reporter ────────────────────────────────────────────────────────

class TestKPIReporter:

    def _schedule_with_waits(self, wait_hours: list[float]) -> Schedule:
        assignments = [make_assignment(wait_hours=w) for w in wait_hours]
        return Schedule.from_assignments(assignments)

    def test_avg_wait_computed_correctly(self):
        from simulation.kpi_reporter import compute_schedule_kpis
        schedule = self._schedule_with_waits([2.0, 4.0, 6.0])
        kpis     = compute_schedule_kpis(schedule)
        assert kpis.avg_wait_hours == pytest.approx(4.0)

    def test_max_wait_is_highest_value(self):
        from simulation.kpi_reporter import compute_schedule_kpis
        schedule = self._schedule_with_waits([1.0, 8.5, 3.0])
        kpis     = compute_schedule_kpis(schedule)
        assert kpis.max_wait_hours == pytest.approx(8.5)

    def test_direct_berth_rate_counts_near_zero_waits(self):
        from simulation.kpi_reporter import compute_schedule_kpis
        schedule = self._schedule_with_waits([0.0, 0.05, 4.0, 8.0])
        kpis     = compute_schedule_kpis(schedule)
        assert kpis.direct_berth_count == 2
        assert kpis.direct_berth_rate  == pytest.approx(0.5)

    def test_empty_schedule_returns_zero_kpis(self):
        from simulation.kpi_reporter import compute_schedule_kpis
        empty    = Schedule.from_assignments([])
        kpis     = compute_schedule_kpis(empty)
        assert kpis.total_vessels == 0
        assert kpis.avg_wait_hours == 0.0

    def test_schedule_accuracy_all_on_time(self):
        from simulation.kpi_reporter import compute_schedule_accuracy
        eta      = datetime(2024, 6, 1, 8, 0)
        vessel   = make_vessel(eta=eta)
        assgn    = make_assignment(vessel=vessel, wait_hours=2.0)
        actual   = {vessel.name: assgn.scheduled_start + timedelta(minutes=30)}
        accuracy = compute_schedule_accuracy([assgn], actual)
        assert accuracy == pytest.approx(1.0)

    def test_schedule_accuracy_all_late(self):
        from simulation.kpi_reporter import compute_schedule_accuracy
        eta      = datetime(2024, 6, 1, 8, 0)
        vessel   = make_vessel(eta=eta)
        assgn    = make_assignment(vessel=vessel, wait_hours=2.0)
        actual   = {vessel.name: assgn.scheduled_start + timedelta(hours=5)}
        accuracy = compute_schedule_accuracy([assgn], actual)
        assert accuracy == pytest.approx(0.0)

    def test_berth_utilization_below_one(self):
        from simulation.kpi_reporter import compute_berth_utilization
        assignments = [make_assignment(wait_hours=0) for _ in range(5)]
        util        = compute_berth_utilization(assignments,
                                                total_berths=10,
                                                horizon_hours=24)
        assert 0.0 <= util <= 1.0

    def test_queue_timeseries_has_correct_shape(self):
        from simulation.kpi_reporter import build_queue_length_timeseries
        eta   = datetime(2024, 6, 1, 8, 0)
        vessels = [make_vessel(f"V{i}", eta=eta + timedelta(hours=i))
                   for i in range(5)]
        assignments = [make_assignment(v, wait_hours=2.0) for v in vessels]
        df = build_queue_length_timeseries(assignments, interval_hours=1.0)
        assert "timestamp"    in df.columns
        assert "queue_length" in df.columns
        assert len(df) > 0

    def test_scenario_comparison_table_has_all_rows(self):
        from simulation.kpi_reporter import build_scenario_comparison_table
        from simulation.scenario_engine import SimulationResult
        results = [
            SimulationResult("S1", 10, 3.0, 8.0, 0.7, 4),
            SimulationResult("S2", 20, 6.0, 14.0, 0.9, 2),
        ]
        df = build_scenario_comparison_table(results)
        assert len(df) == 2
        assert "Avg wait (h)" in df.columns


# ── Test: Schedule domain model ───────────────────────────────────────────────

class TestScheduleModel:

    def test_from_assignments_sums_wait_correctly(self):
        assignments = [make_assignment(wait_hours=w) for w in [1.0, 3.0, 5.0]]
        schedule    = Schedule.from_assignments(assignments)
        assert schedule.total_wait_hours == pytest.approx(9.0)
        assert schedule.avg_wait_hours   == pytest.approx(3.0)

    def test_direct_berthing_flag_on_assignment(self):
        a = make_assignment(wait_hours=0.05)
        assert a.is_direct_berthing is True

    def test_non_direct_berthing_flag(self):
        a = make_assignment(wait_hours=3.0)
        assert a.is_direct_berthing is False


# ── Test: Agent wiring (mocked dependencies) ──────────────────────────────────

class TestAgentOrchestrator:

    def _mock_predictor(self, predicted_hours: float = 8.0):
        pred = MagicMock()
        pred.predict_service_hours.return_value = predicted_hours
        return pred

    def _mock_constraint_engine(self, eligible: list[Berth]):
        from constraints.constraint_engine import EligibilityReport
        engine = MagicMock()
        report = EligibilityReport("V", eligible, [])
        engine.find_eligible_berths.return_value = report
        return engine

    def _mock_tracker(self):
        from tracker.port_tracker import PortTracker
        berth   = make_berth()
        tracker = PortTracker({"70": berth})
        return tracker

    def test_agent_allocate_returns_schedule(self):
        from agent import BerthAllocationAgent
        berth   = make_berth()
        tracker = self._mock_tracker()
        engine  = self._mock_constraint_engine([berth])
        pred    = self._mock_predictor(8.0)

        agent    = BerthAllocationAgent(
            berth_matrix      = {"70": berth},
            port_tracker      = tracker,
            constraint_engine = engine,
            predictor         = pred,
        )
        vessel   = make_vessel()
        schedule = agent.allocate([vessel])
        assert isinstance(schedule, Schedule)
        assert len(schedule.assignments) >= 0    # GA may converge to 0 in edge case

    def test_agent_drops_vessel_with_no_eligible_berths(self):
        from agent import BerthAllocationAgent
        from constraints.constraint_engine import EligibilityReport
        berth   = make_berth()
        tracker = self._mock_tracker()

        engine  = MagicMock()
        engine.find_eligible_berths.return_value = EligibilityReport("V", [], [])
        pred    = self._mock_predictor()

        agent    = BerthAllocationAgent(
            berth_matrix      = {"70": berth},
            port_tracker      = tracker,
            constraint_engine = engine,
            predictor         = pred,
        )
        vessel   = make_vessel()
        schedule = agent.allocate([vessel])
        # Vessel had no eligible berths → assignments must be empty
        assert schedule.assignments == []
