"""
dashboard/simulation_state.py
──────────────────────────────
Mutable simulation state stored in st.session_state.

Responsibilities:
  - Hold the port clock (sim_time)
  - Track which berths are currently occupied (active reservations)
  - Hold the waiting queue
  - Store the full allocation history per time step
  - Advance the clock and update occupancy accordingly

This is the only module allowed to mutate simulation state.
All other modules read from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from config.models import Assignment, Berth, BerthReservation, Schedule, Vessel

# ── Time step sizes ──────────────────────────────────────────────────────────

TIME_STEP_OPTIONS_HOURS = {
    "+1 h":   1,
    "+6 h":   6,
    "+12 h":  12,
    "+24 h":  24,
    "+48 h":  48,
}

DEFAULT_TIME_STEP_HOURS = 24


# ── Active berth slot ─────────────────────────────────────────────────────────

@dataclass
class ActiveBerthSlot:
    """Represents a vessel currently occupying a berth in the simulation."""
    vessel:       Vessel
    berth:        Berth
    arrived_at:   datetime
    departs_at:   datetime

    @property
    def remaining_hours(self) -> float:
        now = SimulationState._current_time_ref
        return max(0.0, (self.departs_at - now).total_seconds() / 3600)

    @property
    def progress_pct(self) -> float:
        total = (self.departs_at - self.arrived_at).total_seconds()
        elapsed = (SimulationState._current_time_ref - self.arrived_at).total_seconds()
        return min(100.0, max(0.0, elapsed / total * 100)) if total > 0 else 0.0

    @property
    def is_complete(self) -> bool:
        return SimulationState._current_time_ref >= self.departs_at


# ── Allocation history entry ───────────────────────────────────────────────────

@dataclass
class AllocationSnapshot:
    """One recorded allocation result at a specific sim_time."""
    sim_time:       datetime
    schedule:       Schedule
    queue_before:   int
    queue_after:    int
    step_number:    int


# ── Simulation state ───────────────────────────────────────────────────────────

@dataclass
class SimulationState:
    """
    Complete mutable state of the port simulation.
    Stored as a single object in st.session_state["sim_state"].
    """

    # Class-level ref used by ActiveBerthSlot.remaining_hours
    _current_time_ref: datetime = field(default_factory=datetime.utcnow)

    # ── Clock ──────────────────────────────────────────────────────────────────
    sim_time:       datetime = field(default_factory=datetime.utcnow)
    step_number:    int = 0

    # ── Port state ─────────────────────────────────────────────────────────────
    active_slots:   list[ActiveBerthSlot]  = field(default_factory=list)
    waiting_queue:  list[Vessel]           = field(default_factory=list)
    berth_matrix:   dict[str, Berth]       = field(default_factory=dict)

    # ── History ────────────────────────────────────────────────────────────────
    allocation_history: list[AllocationSnapshot] = field(default_factory=list)
    kpi_history:        list[dict]               = field(default_factory=list)

    # ── Cumulative KPIs ────────────────────────────────────────────────────────
    total_vessels_served:  int   = 0
    cumulative_wait_hours: float = 0.0
    total_allocations_run: int   = 0

    def __post_init__(self) -> None:
        SimulationState._current_time_ref = self.sim_time

    # ── Clock advancement ──────────────────────────────────────────────────────

    def advance_clock(self, hours: int) -> list[ActiveBerthSlot]:
        """
        Move sim_time forward by `hours`.
        Returns the list of berth slots that completed during this period.
        """
        self.sim_time += timedelta(hours=hours)
        SimulationState._current_time_ref = self.sim_time
        self.step_number += 1

        completed = [s for s in self.active_slots if s.is_complete]
        self.active_slots = [s for s in self.active_slots if not s.is_complete]
        return completed

    # ── Queue management ───────────────────────────────────────────────────────

    def add_vessel_to_queue(self, vessel: Vessel) -> None:
        """Add one vessel to the waiting queue, sorted by ETA."""
        self.waiting_queue.append(vessel)
        self.waiting_queue.sort(key=lambda v: v.eta)

    def remove_vessel_from_queue(self, vessel_name: str) -> None:
        """Remove a vessel from the queue by name."""
        self.waiting_queue = [v for v in self.waiting_queue
                              if v.name != vessel_name]

    # ── Allocation result application ──────────────────────────────────────────

    def apply_schedule(self, schedule: Schedule) -> None:
        """
        Convert a Schedule into ActiveBerthSlots and update all KPIs.
        Removes assigned vessels from the waiting queue.
        """
        queue_before = len(self.waiting_queue)
        assigned_names = {a.vessel.name for a in schedule.assignments}

        for assignment in schedule.assignments:
            slot = ActiveBerthSlot(
                vessel     = assignment.vessel,
                berth      = assignment.berth,
                arrived_at = assignment.scheduled_start,
                departs_at = assignment.predicted_end,
            )
            self.active_slots.append(slot)

        self.waiting_queue = [v for v in self.waiting_queue
                              if v.name not in assigned_names]

        # Update cumulative KPIs
        self.total_vessels_served  += len(schedule.assignments)
        self.cumulative_wait_hours += schedule.total_wait_hours
        self.total_allocations_run += 1

        # Record snapshot
        snapshot = AllocationSnapshot(
            sim_time     = self.sim_time,
            schedule     = schedule,
            queue_before = queue_before,
            queue_after  = len(self.waiting_queue),
            step_number  = self.step_number,
        )
        self.allocation_history.append(snapshot)

        # Record KPI row for charting
        self.kpi_history.append({
            "step":             self.step_number,
            "sim_time":         self.sim_time.strftime("%Y-%m-%d %H:%M"),
            "vessels_served":   len(schedule.assignments),
            "avg_wait_h":       round(schedule.avg_wait_hours, 2),
            "total_wait_h":     round(schedule.total_wait_hours, 2),
            "queue_length":     len(self.waiting_queue),
            "berths_occupied":  len(self.active_slots),
            "utilization_pct":  round(
                len(self.active_slots) / max(1, len(self.berth_matrix)) * 100, 1),
        })

    # ── Computed properties ────────────────────────────────────────────────────

    @property
    def avg_cumulative_wait(self) -> float:
        return (self.cumulative_wait_hours / self.total_vessels_served
                if self.total_vessels_served > 0 else 0.0)

    @property
    def occupied_berth_count(self) -> int:
        return len(self.active_slots)

    @property
    def free_berth_count(self) -> int:
        occupied_ids = {s.berth.berth_id for s in self.active_slots}
        return len(self.berth_matrix) - len(occupied_ids)

    @property
    def utilization_pct(self) -> float:
        return self.occupied_berth_count / max(1, len(self.berth_matrix)) * 100

    def occupied_berth_ids(self) -> set[str]:
        return {s.berth.berth_id for s in self.active_slots}


# ── Factory ───────────────────────────────────────────────────────────────────

def create_initial_state(berth_matrix: dict[str, Berth],
                          initial_active: list[ActiveBerthSlot],
                          initial_queue:  list[Vessel],
                          start_time:     datetime) -> SimulationState:
    """Build the SimulationState from user-configured initial conditions."""
    state = SimulationState(
        sim_time     = start_time,
        berth_matrix = berth_matrix,
        active_slots = initial_active,
        waiting_queue= initial_queue,
    )
    SimulationState._current_time_ref = start_time
    return state
