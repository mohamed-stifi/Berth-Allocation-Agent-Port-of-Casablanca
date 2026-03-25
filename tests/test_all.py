"""
Unit tests for every module in the Berth Allocation Agent.

Design rules:
  - No real files read — all data is created inline
  - No ML training — model is mocked
  - Each test class covers exactly one module
  - Test names describe the behaviour being verified

Run with:  python -m pytest tests/test_all.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.models import Berth, BerthReservation, Vessel


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_vessel(name: str = "VESSEL_A",
                draft_m: float = 9.0,
                length_m: float = 150.0,
                cargo: str = "Containers",
                vessel_type: str = "PORTE_CONTENEUR",
                eta: datetime | None = None) -> Vessel:
    return Vessel(
        name=name, vessel_type=vessel_type, gross_tonnage=20000,
        length_m=length_m, draft_m=draft_m,
        cargo_import=cargo, cargo_export="",
        tonnage_import=5000.0, tonnage_export=0.0,
        tugs_at_berthing=2, arrival_month=6, arrival_dow=1,
        eta=eta or datetime(2024, 6, 1, 8, 0),
    )


def make_berth(berth_id: str = "70",
               depth_m: float = 12.0,
               length_m: float = 600.0,
               cargos: set | None = None) -> Berth:
    return Berth(
        berth_id=berth_id, terminal_name="Marsa Maroc",
        quay_length_m=length_m, depth_m=depth_m,
        allowed_cargos=frozenset(cargos or {"Containers"}),
        allowed_types=frozenset(),
        operator="Marsa Maroc",
    )


# ── Test: BerthReservation overlap logic ──────────────────────────────────────

class TestBerthReservation:
    BASE = datetime(2024, 6, 1, 10, 0)

    def _reservation(self, start_offset_h: int, end_offset_h: int,
                     vessel: str = "V") -> BerthReservation:
        return BerthReservation(
            vessel, "berth_1",
            self.BASE + timedelta(hours=start_offset_h),
            self.BASE + timedelta(hours=end_offset_h),
        )

    def test_overlapping_reservations_detected(self):
        r1 = self._reservation(0, 8)
        r2 = self._reservation(4, 12)
        assert r1.overlaps_with(r2)

    def test_adjacent_reservations_do_not_overlap(self):
        r1 = self._reservation(0, 8)
        r2 = self._reservation(8, 16)
        assert not r1.overlaps_with(r2)

    def test_contained_reservation_overlaps(self):
        outer = self._reservation(0, 24)
        inner = self._reservation(6, 12)
        assert outer.overlaps_with(inner)

    def test_duration_property(self):
        r = self._reservation(0, 6)
        assert r.duration_hours == pytest.approx(6.0)


# ── Test: Constraint Engine ───────────────────────────────────────────────────

class TestConstraintEngine:
    from constraints.constraint_engine import ConstraintEngine

    def _engine(self, berths: list[Berth]) -> ConstraintEngine:
        from constraints.constraint_engine import ConstraintEngine
        matrix = {b.berth_id: b for b in berths}
        return ConstraintEngine(matrix)

    def test_eligible_berth_returned_when_all_constraints_pass(self):
        berth  = make_berth(depth_m=12.0, length_m=600.0, cargos={"Containers"})
        vessel = make_vessel(draft_m=9.0, length_m=150.0, cargo="Containers")
        engine = self._engine([berth])
        report = engine.find_eligible_berths(vessel)
        assert report.eligible_count == 1

    def test_berth_rejected_when_too_shallow(self):
        berth  = make_berth(depth_m=8.0)
        vessel = make_vessel(draft_m=9.0)
        engine = self._engine([berth])
        report = engine.find_eligible_berths(vessel)
        assert report.eligible_count == 0

    def test_berth_rejected_when_too_short(self):
        berth  = make_berth(length_m=100.0)
        vessel = make_vessel(length_m=200.0)
        engine = self._engine([berth])
        report = engine.find_eligible_berths(vessel)
        assert report.eligible_count == 0

    def test_berth_rejected_when_cargo_mismatch(self):
        berth  = make_berth(cargos={"Phosphates"})
        vessel = make_vessel(cargo="Containers")
        engine = self._engine([berth])
        report = engine.find_eligible_berths(vessel)
        assert report.eligible_count == 0

    def test_berth_accepted_when_no_cargo_restriction(self):
        berth  = make_berth(cargos=set())          # empty = accept all
        vessel = make_vessel(cargo="Containers")
        engine = self._engine([berth])
        report = engine.find_eligible_berths(vessel)
        assert report.eligible_count == 1


# ── Test: BerthTimeline ───────────────────────────────────────────────────────

class TestBerthTimeline:
    from tracker.port_tracker import BerthTimeline

    BASE = datetime(2024, 6, 1, 8, 0)

    def _timeline(self) -> "BerthTimeline":
        from tracker.port_tracker import BerthTimeline
        return BerthTimeline(make_berth())

    def test_add_non_overlapping_reservations_succeeds(self):
        tl = self._timeline()
        tl.add_reservation(BerthReservation("V1", "70", self.BASE,
                                             self.BASE + timedelta(hours=8)))
        tl.add_reservation(BerthReservation("V2", "70",
                                             self.BASE + timedelta(hours=8),
                                             self.BASE + timedelta(hours=16)))
        assert tl.current_vessel(self.BASE + timedelta(hours=4)).vessel_name == "V1"

    def test_add_overlapping_reservation_raises(self):
        tl = self._timeline()
        tl.add_reservation(BerthReservation("V1", "70", self.BASE,
                                             self.BASE + timedelta(hours=8)))
        with pytest.raises(ValueError):
            tl.add_reservation(BerthReservation("V2", "70",
                                                  self.BASE + timedelta(hours=4),
                                                  self.BASE + timedelta(hours=12)))

    def test_next_available_slot_skips_occupied_period(self):
        tl = self._timeline()
        tl.add_reservation(BerthReservation("V1", "70", self.BASE,
                                             self.BASE + timedelta(hours=10)))
        next_slot = tl.get_next_available_slot(self.BASE)
        assert next_slot == self.BASE + timedelta(hours=10)

    def test_remove_reservation_frees_berth(self):
        tl = self._timeline()
        tl.add_reservation(BerthReservation("V1", "70", self.BASE,
                                             self.BASE + timedelta(hours=8)))
        tl.remove_reservation("V1")
        assert tl.current_vessel(self.BASE + timedelta(hours=4)) is None


# ── Test: PortTracker ─────────────────────────────────────────────────────────

class TestPortTracker:

    BASE = datetime(2024, 6, 1, 8, 0)

    def _tracker_with_one_berth(self):
        from tracker.port_tracker import PortTracker
        berth  = make_berth()
        return PortTracker({"70": berth}), berth

    def test_assign_then_query_occupied(self):
        tracker, _  = self._tracker_with_one_berth()
        vessel       = make_vessel()
        tracker.assign_vessel_to_berth(vessel, "70", self.BASE,
                                        self.BASE + timedelta(hours=8))
        occupied = tracker.get_occupied_berths(self.BASE + timedelta(hours=4))
        assert "70" in occupied

    def test_release_frees_berth(self):
        tracker, _ = self._tracker_with_one_berth()
        vessel      = make_vessel()
        tracker.assign_vessel_to_berth(vessel, "70", self.BASE,
                                        self.BASE + timedelta(hours=8))
        tracker.release_berth(vessel.name)
        occupied = tracker.get_occupied_berths(self.BASE + timedelta(hours=4))
        assert "70" not in occupied

    def test_late_arrival_detected_above_threshold(self):
        from tracker.port_tracker import PortTracker
        tracker, _ = self._tracker_with_one_berth()
        vessel      = make_vessel(eta=self.BASE)
        actual      = self.BASE + timedelta(hours=4)   # 4 h late > 2 h threshold
        assert tracker.detect_late_arrival(vessel, actual) is True

    def test_minor_delay_not_flagged_as_late(self):
        from tracker.port_tracker import PortTracker
        tracker, _ = self._tracker_with_one_berth()
        vessel      = make_vessel(eta=self.BASE)
        actual      = self.BASE + timedelta(hours=1)   # only 1 h < threshold
        assert tracker.detect_late_arrival(vessel, actual) is False

    def test_invalid_berth_raises_key_error(self):
        from tracker.port_tracker import PortTracker
        tracker, _ = self._tracker_with_one_berth()
        with pytest.raises(KeyError):
            tracker.get_next_available_slot("NONEXISTENT", self.BASE)


# ── Test: Greedy dispatch ─────────────────────────────────────────────────────

class TestGreedyDispatch:

    BASE = datetime(2024, 6, 1, 8, 0)

    def test_assigns_each_vessel_to_one_berth(self):
        from optimizer.genetic_algorithm import greedy_dispatch
        from tracker.port_tracker import PortTracker

        berth_a    = make_berth("70", depth_m=12, length_m=600)
        berth_b    = make_berth("30", depth_m=10, length_m=500,
                                cargos={"General Cargo"})
        vessel_a   = make_vessel("VA", eta=self.BASE)
        vessel_b   = make_vessel("VB", eta=self.BASE + timedelta(hours=2))

        tracker    = PortTracker({"70": berth_a, "30": berth_b})
        eligible   = {"VA": [berth_a], "VB": [berth_a]}
        svc_times  = {"VA": 8.0, "VB": 6.0}

        chromosome = greedy_dispatch([vessel_a, vessel_b],
                                      eligible, svc_times, tracker)
        assert len(chromosome) == 2

    def test_greedy_respects_vessel_eta(self):
        from optimizer.genetic_algorithm import greedy_dispatch
        from tracker.port_tracker import PortTracker

        berth   = make_berth("70", depth_m=12, length_m=600)
        future  = self.BASE + timedelta(hours=5)
        vessel  = make_vessel("VF", eta=future)
        tracker = PortTracker({"70": berth})
        eligible= {"VF": [berth]}
        svc     = {"VF": 4.0}

        chromosome = greedy_dispatch([vessel], eligible, svc, tracker)
        _, _, start = chromosome[0]
        assert start >= future


# ── Test: Data pipeline (no file I/O) ────────────────────────────────────────

class TestDataPipeline:
    import pandas as pd

    def _minimal_df(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame({
            "Date Entrée au Port":  ["01/06/2024"],
            "Heure Entrée au Port": ["08:00"],
            "Date Accostage":       ["01/06/2024"],
            "Heure Accostage":      ["10:00"],
            "Date App Quai":        ["02/06/2024"],
            "Heure App Quai":       ["08:00"],
            "Date Entree Mouillage":["01/06/2024"],
            "Poste Accostage":      [" 70 "],
        })

    def test_parse_event_datetimes_adds_three_columns(self):
        from pipeline.data_pipeline import parse_event_datetimes
        df     = self._minimal_df()
        result = parse_event_datetimes(df)
        assert "arrival_datetime"    in result.columns
        assert "berthing_datetime"   in result.columns
        assert "unberthing_datetime" in result.columns

    def test_waiting_time_is_non_negative(self):
        from pipeline.data_pipeline import parse_event_datetimes, derive_waiting_time
        df     = parse_event_datetimes(self._minimal_df())
        result = derive_waiting_time(df)
        assert (result["waiting_time_hours"] >= 0).all()

    def test_anchorage_flag_set_when_anchorage_date_present(self):
        from pipeline.data_pipeline import flag_anchorage_usage
        df     = self._minimal_df()
        result = flag_anchorage_usage(df)
        assert result["anchored_before_berthing"].iloc[0] == 1

    def test_normalize_berth_ids_strips_whitespace(self):
        from pipeline.data_pipeline import normalize_berth_ids
        df     = self._minimal_df()
        result = normalize_berth_ids(df)
        assert result["Poste Accostage"].iloc[0] == "70"
