"""
Layer 5A — Simulation Engine (SimPy discrete-event simulation).

Responsibility: replay historical data or inject synthetic events to
validate the Berth Allocation Agent under controlled scenarios.

Seven scenarios:
  S1 — Baseline replay        (historical vessel queue)
  S2 — High congestion        (2× arrival rate)
  S3 — Weather port closure   (port shut N hours then reopened)
  S4 — Berth maintenance      (N berths taken offline)
  S5 — Late vessel arrival    (+6 h delay on one vessel)
  S6 — Priority vessel        (one vessel gets queue priority)
  S7 — Multi-terminal conflict(two large ships, one slot each)

Each scenario function returns a SimulationResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import simpy

from config.constants import (
    SIM_CONGESTION_MULTIPLIER, SIM_LATE_ETA_DELAY_HOURS,
    SIM_MAINTENANCE_BERTHS, SIM_PRIORITY_FLAG, SIM_WEATHER_CLOSURE_HOURS,
)
from config.models import Assignment, Berth, Schedule, Vessel
from tracker.port_tracker import PortTracker

logger = logging.getLogger(__name__)

SIMULATION_TIME_UNIT_HOURS = 1.0   # 1 SimPy time unit = 1 real hour


def _group_berths_by_terminal(berth_matrix: dict[str, Berth]
                              ) -> dict[str, list[Berth]]:
    """Partition berth matrix into terminal-named groups."""
    groups: dict[str, list[Berth]] = {}
    for berth in berth_matrix.values():
        groups.setdefault(berth.terminal_name, []).append(berth)
    return groups


def _find_vessel_terminal(vessel: Vessel,
                          terminal_berths: dict[str, list[Berth]]
                          ) -> str | None:
    """Return the terminal name where this vessel has at least one berth
    that satisfies ALL hard constraints (depth, length, cargo, type)."""
    from constraints.constraint_engine import _berth_depth_fits_vessel
    from constraints.constraint_engine import _berth_length_fits_vessel
    from constraints.constraint_engine import _berth_accepts_cargo
    from constraints.constraint_engine import _berth_accepts_vessel_type

    for term_name, berths in terminal_berths.items():
        if any(
            _berth_depth_fits_vessel(b, vessel) and
            _berth_length_fits_vessel(b, vessel) and
            _berth_accepts_cargo(b, vessel) and
            _berth_accepts_vessel_type(b, vessel)
            for b in berths
        ):
            return term_name
    return None


@dataclass
class SimulationResult:
    """Holds KPIs computed after one simulation run."""
    scenario_name:        str
    total_vessels_served: int
    avg_waiting_hours:    float
    max_waiting_hours:    float
    berth_utilization:    float
    vessels_with_zero_wait: int
    assignments:          list[Assignment] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.scenario_name}] "
            f"served={self.total_vessels_served}  "
            f"avg_wait={self.avg_waiting_hours:.1f}h  "
            f"max_wait={self.max_waiting_hours:.1f}h  "
            f"utilization={self.berth_utilization:.1%}  "
            f"direct_berth={self.vessels_with_zero_wait}"
        )


# ── SimPy process helpers ─────────────────────────────────────────────────────

def _vessel_arrival_process(env: simpy.Environment,
                             vessel: Vessel,
                             berth_resource: simpy.Resource,
                             service_hours: float,
                             results: list[Assignment],
                             epoch: datetime) -> None:
    """SimPy generator: vessel arrives, waits for a berth, occupies it, leaves."""
    arrival_sim_time = (vessel.eta - epoch).total_seconds() / 3600
    yield env.timeout(max(0, arrival_sim_time - env.now))

    request_time = env.now
    with berth_resource.request() as req:
        yield req
        wait_hours   = env.now - request_time
        start_dt     = epoch + timedelta(hours=env.now)
        end_dt       = start_dt + timedelta(hours=service_hours)
        results.append(Assignment(
            vessel               = vessel,
            berth                = Berth("simulated", "sim", 999, 99,
                                         frozenset(), frozenset(), "sim"),
            scheduled_start      = start_dt,
            predicted_end        = end_dt,
            predicted_wait_hours = wait_hours,
        ))
        yield env.timeout(service_hours)


def _run_simpy_scenario(scenario_name: str,
                          vessels: list[Vessel],
                          service_times: dict[str, float],
                          berth_capacity: int,
                          berth_matrix: dict[str, Berth] | None = None
                          ) -> SimulationResult:
    """Execute a SimPy simulation — per-terminal pools when berth_matrix is provided."""
    if not vessels:
        return _compute_result(scenario_name, [], berth_capacity, berth_matrix)

    if berth_matrix:
        return _run_terminal_stratified_scenario(scenario_name, vessels,
                                                  service_times, berth_matrix)

    epoch          = min(v.eta for v in vessels)
    last_eta       = max(v.eta for v in vessels)
    max_service    = max(service_times.get(v.name, 12.0) for v in vessels)
    horizon_hours  = ((last_eta - epoch).total_seconds() / 3600) + max_service + 24

    env          = simpy.Environment()
    berth_pool   = simpy.Resource(env, capacity=berth_capacity)
    assignments: list[Assignment] = []

    for vessel in vessels:
        env.process(_vessel_arrival_process(
            env, vessel, berth_pool, service_times.get(vessel.name, 12.0),
            assignments, epoch))

    env.run(until=horizon_hours)

    return _compute_result(scenario_name, assignments, berth_capacity, berth_matrix)


def _run_terminal_stratified_scenario(scenario_name: str,
                                      vessels: list[Vessel],
                                      service_times: dict[str, float],
                                      berth_matrix: dict[str, Berth]
                                      ) -> SimulationResult:
    """Run one SimPy sim per terminal so vessels only compete within their terminal."""
    terminal_berths = _group_berths_by_terminal(berth_matrix)
    all_assignments: list[Assignment] = []
    total_capacity  = 0
    all_vessels: list[Vessel] = []
    epoch      = min(v.eta for v in vessels)
    last_eta   = max(v.eta for v in vessels)
    max_svc    = max(service_times.get(v.name, 12.0) for v in vessels)
    horizon    = ((last_eta - epoch).total_seconds() / 3600) + max_svc + 24

    for term_name, berths in terminal_berths.items():
        term_capacity = len(berths)
        total_capacity += term_capacity
        term_vessels = [v for v in vessels
                        if _find_vessel_terminal(v, terminal_berths) == term_name]
        all_vessels.extend(term_vessels)

        if not term_vessels:
            continue

        env    = simpy.Environment()
        pool   = simpy.Resource(env, capacity=term_capacity)

        for vessel in term_vessels:
            env.process(_vessel_arrival_process(
                env, vessel, pool, service_times.get(vessel.name, 12.0),
                all_assignments, epoch))
        env.run(until=horizon)

    return _compute_result(scenario_name, all_assignments, total_capacity, berth_matrix)


def _compute_result(scenario_name: str,
                    assignments: list[Assignment],
                    berth_capacity: int,
                    berth_matrix: dict[str, Berth] | None = None
                    ) -> SimulationResult:
    """Aggregate assignment list into a SimulationResult KPI object."""
    if not assignments:
        return SimulationResult(scenario_name, 0, 0.0, 0.0, 0.0, 0)

    wait_times    = [a.predicted_wait_hours for a in assignments]
    total_service = sum((a.predicted_end - a.scheduled_start).total_seconds()
                        for a in assignments) / 3600
    horizon_hours = max(
        (a.predicted_end - assignments[0].scheduled_start).total_seconds() / 3600
        for a in assignments) if assignments else 1.0

    if berth_matrix:
        occupied_terminals: set[str] = set()
        for a in assignments:
            occupied_terminals.add(a.berth.terminal_name)
        effective_capacity = sum(
            len([b for b in berth_matrix.values() if b.terminal_name == t])
            for t in occupied_terminals
        )
        effective_capacity = effective_capacity or berth_capacity
    else:
        effective_capacity = berth_capacity

    return SimulationResult(
        scenario_name          = scenario_name,
        total_vessels_served   = len(assignments),
        avg_waiting_hours      = sum(wait_times) / len(wait_times),
        max_waiting_hours      = max(wait_times),
        berth_utilization      = total_service / (effective_capacity * horizon_hours)
                                if effective_capacity and horizon_hours else 0.0,
        vessels_with_zero_wait = sum(1 for w in wait_times if w < 0.1),
        assignments            = assignments,
    )


# ── Scenario S1 — Baseline replay ─────────────────────────────────────────────

def scenario_baseline_replay(historical_vessels: list[Vessel],
                              service_times: dict[str, float],
                              total_berths: int,
                              berth_matrix: dict[str, Berth] | None = None
                              ) -> SimulationResult:
    """Replay historical vessel arrivals in order — ground truth comparison."""
    logger.info("Running S1 — Baseline replay (%d vessels)", len(historical_vessels))
    sorted_vessels = sorted(historical_vessels, key=lambda v: v.eta)
    return _run_simpy_scenario("S1_Baseline", sorted_vessels,
                                service_times, total_berths, berth_matrix)


# ── Scenario S2 — High congestion ─────────────────────────────────────────────

def _duplicate_vessels_at_compressed_interval(vessels: list[Vessel]) -> list[Vessel]:
    """Clone vessel list with halved inter-arrival gaps (= 2× arrival rate)."""
    if len(vessels) < 2:
        return vessels
    augmented = list(vessels)
    for i in range(len(vessels) - 1):
        gap        = vessels[i + 1].eta - vessels[i].eta
        midpoint   = vessels[i].eta + gap / 2
        extra      = Vessel(**{**vessels[i].__dict__, "eta": midpoint,
                               "name": vessels[i].name + "_EXTRA"})
        augmented.append(extra)
    return sorted(augmented, key=lambda v: v.eta)


def scenario_high_congestion(historical_vessels: list[Vessel],
                              service_times: dict[str, float],
                              total_berths: int,
                              berth_matrix: dict[str, Berth] | None = None
                              ) -> SimulationResult:
    """Double the arrival rate to stress-test queue management."""
    logger.info("Running S2 — High congestion (%.0f× rate)", SIM_CONGESTION_MULTIPLIER)
    crowded = _duplicate_vessels_at_compressed_interval(historical_vessels)
    extra_service = {v.name: service_times.get(
        v.name.replace("_EXTRA", ""), 12.0) for v in crowded}
    return _run_simpy_scenario("S2_Congestion", crowded, extra_service,
                               total_berths, berth_matrix)


# ── Scenario S3 — Weather port closure ────────────────────────────────────────

def scenario_weather_closure(vessels: list[Vessel],
                              service_times: dict[str, float],
                              closure_start: datetime,
                              total_berths: int,
                              berth_matrix: dict[str, Berth] | None = None
                              ) -> SimulationResult:
    """
    Hold all arrivals during a closure window, then release the built-up queue.
    Simulates the burst effect after a storm.
    """
    logger.info("Running S3 — Weather closure (%d h)", SIM_WEATHER_CLOSURE_HOURS)
    closure_end = closure_start + timedelta(hours=SIM_WEATHER_CLOSURE_HOURS)

    delayed_vessels = []
    for vessel in vessels:
        if closure_start <= vessel.eta < closure_end:
            delayed_eta = closure_end + timedelta(minutes=10)
            delayed_vessels.append(
                Vessel(**{**vessel.__dict__, "eta": delayed_eta}))
        else:
            delayed_vessels.append(vessel)

    return _run_simpy_scenario("S3_WeatherClosure",
                                sorted(delayed_vessels, key=lambda v: v.eta),
                                service_times, total_berths, berth_matrix)


# ── Scenario S4 — Berth maintenance ───────────────────────────────────────────

def scenario_berth_maintenance(vessels: list[Vessel],
                                service_times: dict[str, float],
                                total_berths: int,
                                berth_matrix: dict[str, Berth] | None = None
                                ) -> SimulationResult:
    """Remove berths from service to test constraint flexibility."""
    offline_count   = len(SIM_MAINTENANCE_BERTHS)
    reduced_berths  = max(1, total_berths - offline_count)
    logger.info("Running S4 — Maintenance: %d berths offline → %d remaining",
                offline_count, reduced_berths)

    maint_matrix = None
    if berth_matrix:
        maint_matrix = {k: v for k, v in berth_matrix.items()
                        if k not in SIM_MAINTENANCE_BERTHS}

    return _run_simpy_scenario("S4_Maintenance", vessels,
                                service_times, reduced_berths, maint_matrix)


# ── Scenario S5 — Late vessel arrival ────────────────────────────────────────

def scenario_late_arrival(vessels: list[Vessel],
                           service_times: dict[str, float],
                           late_vessel_name: str,
                           total_berths: int,
                           berth_matrix: dict[str, Berth] | None = None
                           ) -> SimulationResult:
    """Delay one vessel's arrival by SIM_LATE_ETA_DELAY_HOURS and replan."""
    logger.info("Running S5 — Late arrival: %s delayed +%d h",
                late_vessel_name, SIM_LATE_ETA_DELAY_HOURS)
    modified = []
    for vessel in vessels:
        if vessel.name == late_vessel_name:
            new_eta = vessel.eta + timedelta(hours=SIM_LATE_ETA_DELAY_HOURS)
            modified.append(Vessel(**{**vessel.__dict__, "eta": new_eta}))
        else:
            modified.append(vessel)
    return _run_simpy_scenario("S5_LateArrival",
                                sorted(modified, key=lambda v: v.eta),
                                service_times, total_berths, berth_matrix)


# ── Scenario S6 — Priority vessel ─────────────────────────────────────────────

def scenario_priority_vessel(vessels: list[Vessel],
                              service_times: dict[str, float],
                              priority_vessel_name: str,
                              total_berths: int,
                              berth_matrix: dict[str, Berth] | None = None
                              ) -> SimulationResult:
    """
    Move one vessel to the front of the queue regardless of ETA.
    Simulates urgent cargo (grains, perishables, military).
    """
    logger.info("Running S6 — Priority vessel: %s", priority_vessel_name)
    priority = [v for v in vessels if v.name == priority_vessel_name]
    rest     = [v for v in vessels if v.name != priority_vessel_name]
    reordered = priority + sorted(rest, key=lambda v: v.eta)
    return _run_simpy_scenario("S6_Priority", reordered,
                                service_times, total_berths, berth_matrix)


# ── Scenario S7 — Multi-terminal conflict ─────────────────────────────────────

def scenario_multi_terminal_conflict(vessels: list[Vessel],
                                     service_times: dict[str, float],
                                     total_berths: int,
                                     berth_matrix: dict[str, Berth] | None = None
                                     ) -> SimulationResult:
    """
    Simulate two large vessels arriving simultaneously with only one slot
    available per terminal — tests inter-terminal coordination.
    """
    logger.info("Running S7 — Multi-terminal conflict")
    if len(vessels) < 2:
        return _run_simpy_scenario("S7_Conflict", vessels,
                                    service_times, total_berths, berth_matrix)

    reference_eta = vessels[0].eta
    conflicted    = []
    for vessel in vessels[:2]:
        conflicted.append(Vessel(**{**vessel.__dict__, "eta": reference_eta}))
    conflicted   += vessels[2:]

    conflict_matrix = None
    if berth_matrix and len(vessels) >= 2:
        conflict_matrix = {k: v for k, v in berth_matrix.items()}

    return _run_simpy_scenario("S7_Conflict",
                                sorted(conflicted, key=lambda v: v.eta),
                                service_times,
                                max(1, total_berths - 1),
                                conflict_matrix)


# ── Run all scenarios and compare ────────────────────────────────────────────

def run_all_scenarios(historical_vessels: list[Vessel],
                      service_times: dict[str, float],
                      total_berths: int,
                      closure_start: datetime,
                      berth_matrix: dict[str, Berth] | None = None
                      ) -> list[SimulationResult]:
    """Execute all 7 scenarios and return results for comparison."""
    late_vessel     = historical_vessels[5].name if len(historical_vessels) > 5 else historical_vessels[0].name
    priority_vessel = historical_vessels[0].name

    results = [
        scenario_baseline_replay(historical_vessels, service_times, total_berths, berth_matrix),
        scenario_high_congestion(historical_vessels, service_times, total_berths, berth_matrix),
        scenario_weather_closure(historical_vessels, service_times, closure_start, total_berths, berth_matrix),
        scenario_berth_maintenance(historical_vessels, service_times, total_berths, berth_matrix),
        scenario_late_arrival(historical_vessels, service_times, late_vessel, total_berths, berth_matrix),
        scenario_priority_vessel(historical_vessels, service_times, priority_vessel, total_berths, berth_matrix),
        scenario_multi_terminal_conflict(historical_vessels, service_times, total_berths, berth_matrix),
    ]

    for result in results:
        logger.info(result.summary())

    return results
