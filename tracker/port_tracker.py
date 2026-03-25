"""
Layer 4 — Port Tracker (State Manager).

Responsibility: maintain the current and future state of every
berth in the port. It is the system's memory.

Functions:
  - assign a vessel to a berth for a time window
  - release a berth when a vessel departs
  - query next available slot on any berth
  - detect replan triggers
  - produce port snapshots for dashboard and optimizer feedback

SOLID: Single responsibility — state only, no optimization.
"""

from __future__ import annotations

import logging
from bisect import insort
from datetime import datetime, timedelta

from config.constants import LATE_ARRIVAL_THRESHOLD_HOURS
from config.models import Berth, BerthReservation, PortSnapshot, Vessel

logger = logging.getLogger(__name__)


class BerthTimeline:
    """
    Ordered list of non-overlapping reservations for a single berth.
    Invariant: reservations are always sorted by start_time.
    """

    def __init__(self, berth: Berth) -> None:
        self.berth: Berth                        = berth
        self._reservations: list[BerthReservation] = []

    def add_reservation(self, reservation: BerthReservation) -> None:
        """Insert a reservation in chronological order (fast binary insert)."""
        if self._has_overlap(reservation):
            raise ValueError(
                f"Berth {self.berth.berth_id}: "
                f"reservation for {reservation.vessel_name} overlaps existing booking."
            )
        insort(self._reservations, reservation,
               key=lambda r: r.start_time)

    def remove_reservation(self, vessel_name: str) -> bool:
        """Remove the reservation for a vessel. Returns True if found."""
        before = len(self._reservations)
        self._reservations = [r for r in self._reservations
                              if r.vessel_name != vessel_name]
        return len(self._reservations) < before

    def get_next_available_slot(self, not_before: datetime) -> datetime:
        """Return the earliest datetime this berth is free after `not_before`."""
        candidate = not_before
        for reservation in self._reservations:
            if reservation.start_time > candidate + timedelta(minutes=1):
                break
            if reservation.end_time > candidate:
                candidate = reservation.end_time
        return candidate

    def is_available_during(self, start: datetime, end: datetime) -> bool:
        """True if no existing reservation overlaps the proposed window."""
        probe = BerthReservation("probe", self.berth.berth_id, start, end)
        return not self._has_overlap(probe)

    def current_vessel(self, at_time: datetime) -> BerthReservation | None:
        """Return the active reservation at `at_time`, or None if free."""
        for reservation in self._reservations:
            if reservation.start_time <= at_time < reservation.end_time:
                return reservation
        return None

    def _has_overlap(self, candidate: BerthReservation) -> bool:
        return any(candidate.overlaps_with(r) for r in self._reservations)


class PortTracker:
    """
    Central state manager for all berths in the port.
    Exposes assign / release / query / snapshot / replan-trigger API.
    """

    def __init__(self, berth_matrix: dict[str, Berth]) -> None:
        self._timelines: dict[str, BerthTimeline] = {
            bid: BerthTimeline(berth) for bid, berth in berth_matrix.items()
        }
        self._vessel_registry: dict[str, str] = {}   # vessel_name → berth_id

    # ── Write operations ──────────────────────────────────────────────────────

    def assign_vessel_to_berth(self, vessel: Vessel, berth_id: str,
                                start_time: datetime, end_time: datetime) -> None:
        """Block a time slot on the berth and register the vessel."""
        self._validate_berth_exists(berth_id)
        reservation = BerthReservation(vessel.name, berth_id, start_time, end_time)
        self._timelines[berth_id].add_reservation(reservation)
        self._vessel_registry[vessel.name] = berth_id
        logger.info("Assigned %s → berth %s  [%s – %s]",
                    vessel.name, berth_id,
                    start_time.strftime("%Y-%m-%d %H:%M"),
                    end_time.strftime("%Y-%m-%d %H:%M"))

    def release_berth(self, vessel_name: str) -> None:
        """Mark a berth as free when a vessel departs."""
        berth_id = self._vessel_registry.pop(vessel_name, None)
        if berth_id is None:
            logger.warning("release_berth: vessel '%s' not found in registry", vessel_name)
            return
        found = self._timelines[berth_id].remove_reservation(vessel_name)
        if found:
            logger.info("Released berth %s from %s", berth_id, vessel_name)

    def take_berth_offline(self, berth_id: str, from_time: datetime,
                           until_time: datetime, reason: str) -> None:
        """Block a berth for maintenance or weather closure."""
        self._validate_berth_exists(berth_id)
        maintenance = BerthReservation("__OFFLINE__", berth_id, from_time, until_time)
        self._timelines[berth_id].add_reservation(maintenance)
        logger.warning("Berth %s taken OFFLINE [%s – %s]: %s",
                       berth_id, from_time, until_time, reason)

    # ── Query operations ──────────────────────────────────────────────────────

    def get_next_available_slot(self, berth_id: str, not_before: datetime) -> datetime:
        """Return when the berth is next free after the given time."""
        self._validate_berth_exists(berth_id)
        return self._timelines[berth_id].get_next_available_slot(not_before)

    def is_berth_available(self, berth_id: str,
                           start: datetime, end: datetime) -> bool:
        """True if the berth has no conflicts in the proposed window."""
        self._validate_berth_exists(berth_id)
        return self._timelines[berth_id].is_available_during(start, end)

    def get_occupied_berths(self, at_time: datetime) -> dict[str, BerthReservation]:
        """Return a snapshot of all berths currently occupied at `at_time`."""
        return {
            bid: timeline.current_vessel(at_time)
            for bid, timeline in self._timelines.items()
            if timeline.current_vessel(at_time) is not None
        }

    def get_free_berth_ids(self, at_time: datetime) -> list[str]:
        """Return IDs of all berths that are free at `at_time`."""
        return [bid for bid, tl in self._timelines.items()
                if tl.current_vessel(at_time) is None]

    # ── Replan trigger detection ──────────────────────────────────────────────

    def detect_late_arrival(self, vessel: Vessel, actual_arrival: datetime) -> bool:
        """
        Return True if a vessel arrived significantly later than its ETA,
        signalling that the current schedule should be re-optimized.
        """
        delay_hours = (actual_arrival - vessel.eta).total_seconds() / 3600
        is_late     = delay_hours > LATE_ARRIVAL_THRESHOLD_HOURS
        if is_late:
            logger.warning("Late arrival: %s delayed by %.1f h → replan triggered",
                           vessel.name, delay_hours)
        return is_late

    # ── Snapshot for dashboard and optimizer feedback ─────────────────────────

    def get_port_snapshot(self, at_time: datetime,
                          vessel_queue: list[Vessel]) -> PortSnapshot:
        """Produce a full port state snapshot at the given moment."""
        occupied = self.get_occupied_berths(at_time)
        free     = self.get_free_berth_ids(at_time)
        return PortSnapshot(
            timestamp       = at_time,
            occupied_berths = occupied,
            free_berths     = free,
            vessel_queue    = vessel_queue,
            queue_length    = len(vessel_queue),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_berth_exists(self, berth_id: str) -> None:
        if berth_id not in self._timelines:
            raise KeyError(f"Berth '{berth_id}' is not in the port's berth matrix.")
