"""
Layer 3 — Berth Allocation Optimizer (Genetic Algorithm).

Responsibility: given a queue of vessels with predicted service times
and their eligible berth sets, find the assignment schedule that
minimizes total anchorage waiting time.

Structure:
  - Chromosome  : list of (vessel_index, berth_id, start_time) tuples
  - Population  : list of chromosomes
  - Fitness     : 1 / (1 + total_waiting_hours) — higher is better
  - Warm-start  : greedy_dispatch builds the initial best chromosome
  - Repair      : fix constraint violations after crossover / mutation

SOLID: The GA is fully contained here. Port Tracker and Constraint
Engine are injected — no direct imports of their internals.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from config.constants import (
    GA_CROSSOVER_PROBABILITY, GA_MAX_GENERATIONS, GA_MUTATION_PROBABILITY,
    GA_NO_IMPROVE_PATIENCE, GA_POPULATION_SIZE, GA_TOURNAMENT_SIZE,
    RANDOM_STATE,
)
from config.models import Assignment, Berth, Schedule, Vessel
from tracker.port_tracker import PortTracker

logger = logging.getLogger(__name__)

random.seed(RANDOM_STATE)

# ── Types ─────────────────────────────────────────────────────────────────────

Gene        = tuple[int, str, datetime]    # (vessel_index, berth_id, start_time)
Chromosome  = list[Gene]
Population  = list[Chromosome]

ServiceTimeMap  = dict[str, float]         # vessel_name → predicted hours
EligibleBerthMap= dict[str, list[Berth]]   # vessel_name → eligible Berth list


# ── Fitness ───────────────────────────────────────────────────────────────────

def compute_total_waiting_hours(chromosome: Chromosome,
                                vessels: list[Vessel]) -> float:
    """Sum of (scheduled_start - ETA) for all vessels in the chromosome."""
    total = 0.0
    for vessel_index, _, start_time in chromosome:
        eta            = vessels[vessel_index].eta
        waiting_hours  = max(0.0, (start_time - eta).total_seconds() / 3600)
        total         += waiting_hours
    return total


def compute_fitness(chromosome: Chromosome, vessels: list[Vessel]) -> float:
    """Higher fitness = lower total waiting time. Bounded in (0, 1]."""
    total_wait = compute_total_waiting_hours(chromosome, vessels)
    return 1.0 / (1.0 + total_wait)


# ── Greedy warm-start ─────────────────────────────────────────────────────────

def greedy_dispatch(vessels: list[Vessel],
                    eligible_berths: EligibleBerthMap,
                    service_times: ServiceTimeMap,
                    tracker: PortTracker) -> Chromosome:
    """
    Assign each vessel (sorted by ETA) to the earliest-available eligible berth.
    Produces a valid but suboptimal chromosome that seeds the GA population.
    """
    berth_busy_until: dict[str, datetime] = {}
    chromosome: Chromosome = []

    for idx, vessel in enumerate(sorted(vessels, key=lambda v: v.eta)):
        best_berth, best_start = _find_earliest_eligible_slot(
            vessel, eligible_berths[vessel.name], berth_busy_until, tracker)

        if best_berth is None:
            logger.warning("Greedy: no berth found for %s — skipping", vessel.name)
            continue

        duration_hours = service_times[vessel.name]
        end_time       = best_start + timedelta(hours=duration_hours)
        berth_busy_until[best_berth.berth_id] = end_time

        original_index = vessels.index(vessel)
        chromosome.append((original_index, best_berth.berth_id, best_start))

    return chromosome


def _find_earliest_eligible_slot(vessel: Vessel,
                                  eligible: list[Berth],
                                  berth_busy_until: dict[str, datetime],
                                  tracker: PortTracker) -> tuple[Berth | None, datetime]:
    """Return (berth, earliest_start) for the berth that minimises wait."""
    best_berth: Berth | None = None
    best_start: datetime     = datetime.max

    for berth in eligible:
        not_before  = max(vessel.eta, berth_busy_until.get(berth.berth_id, vessel.eta))
        slot_start  = tracker.get_next_available_slot(berth.berth_id, not_before)
        if slot_start < best_start:
            best_start = slot_start
            best_berth = berth

    return best_berth, best_start


# ── Population initialization ─────────────────────────────────────────────────

def _random_chromosome(vessels: list[Vessel],
                        eligible_berths: EligibleBerthMap,
                        service_times: ServiceTimeMap,
                        tracker: PortTracker) -> Chromosome:
    """
    Build a random valid chromosome by shuffling vessel order
    and picking a random eligible berth for each.
    """
    order         = list(range(len(vessels)))
    random.shuffle(order)
    berth_busy_until: dict[str, datetime] = {}
    chromosome: Chromosome = []

    for idx in order:
        vessel    = vessels[idx]
        eligible  = eligible_berths[vessel.name]
        if not eligible:
            continue
        berth       = random.choice(eligible)
        not_before  = max(vessel.eta, berth_busy_until.get(berth.berth_id, vessel.eta))
        start_time  = tracker.get_next_available_slot(berth.berth_id, not_before)
        duration    = service_times[vessel.name]
        berth_busy_until[berth.berth_id] = start_time + timedelta(hours=duration)
        chromosome.append((idx, berth.berth_id, start_time))

    return chromosome


def initialize_population(vessels: list[Vessel],
                           eligible_berths: EligibleBerthMap,
                           service_times: ServiceTimeMap,
                           tracker: PortTracker,
                           greedy_seed: Chromosome) -> Population:
    """
    Create the initial population.
    First chromosome is the greedy seed; rest are random.
    """
    population: Population = [greedy_seed]
    while len(population) < GA_POPULATION_SIZE:
        chromosome = _random_chromosome(vessels, eligible_berths, service_times, tracker)
        population.append(chromosome)
    return population


# ── Selection ─────────────────────────────────────────────────────────────────

def tournament_selection(population: Population,
                          fitness_fn: Callable[[Chromosome], float]) -> Chromosome:
    """Pick the best chromosome out of a random tournament subset."""
    contestants = random.sample(population, min(GA_TOURNAMENT_SIZE, len(population)))
    return max(contestants, key=fitness_fn)


# ── Crossover ─────────────────────────────────────────────────────────────────

def order_based_crossover(parent_a: Chromosome, parent_b: Chromosome) -> Chromosome:
    """
    Order-based crossover (OX): inherit a random slice from parent_a,
    fill remaining positions in parent_b order (preserves permutation).
    """
    if len(parent_a) < 2:
        return copy.deepcopy(parent_a)

    cut_lo = random.randint(0, len(parent_a) - 1)
    cut_hi = random.randint(cut_lo + 1, len(parent_a))
    child  = parent_a[cut_lo:cut_hi]

    in_child_indices = {gene[0] for gene in child}
    for gene in parent_b:
        if gene[0] not in in_child_indices:
            child.append(gene)
    return child


# ── Mutation ──────────────────────────────────────────────────────────────────

def _swap_berth_assignment(chromosome: Chromosome,
                            vessels: list[Vessel],
                            eligible_berths: EligibleBerthMap) -> Chromosome:
    """Reassign one random vessel to a different eligible berth."""
    mutant = copy.deepcopy(chromosome)
    if not mutant:
        return mutant

    pos               = random.randrange(len(mutant))
    vessel_index, _, start_time = mutant[pos]
    vessel            = vessels[vessel_index]
    eligible          = eligible_berths.get(vessel.name, [])
    if len(eligible) > 1:
        new_berth     = random.choice([b for b in eligible
                                       if b.berth_id != mutant[pos][1]])
        mutant[pos]   = (vessel_index, new_berth.berth_id, start_time)
    return mutant


def _shift_start_time(chromosome: Chromosome) -> Chromosome:
    """Nudge one vessel's start time by a random offset within ±3 hours."""
    mutant = copy.deepcopy(chromosome)
    if not mutant:
        return mutant

    pos                         = random.randrange(len(mutant))
    vessel_index, berth_id, st  = mutant[pos]
    shift_hours                 = random.uniform(-3.0, 3.0)
    new_start                   = st + timedelta(hours=shift_hours)
    mutant[pos]                 = (vessel_index, berth_id, new_start)
    return mutant


def mutate(chromosome: Chromosome,
           vessels: list[Vessel],
           eligible_berths: EligibleBerthMap) -> Chromosome:
    """Apply one of two mutation operators at random."""
    if random.random() < 0.5:
        return _swap_berth_assignment(chromosome, vessels, eligible_berths)
    return _shift_start_time(chromosome)


# ── Repair ────────────────────────────────────────────────────────────────────

def repair_chromosome(chromosome: Chromosome,
                       vessels: list[Vessel],
                       eligible_berths: EligibleBerthMap,
                       service_times: ServiceTimeMap) -> Chromosome:
    """
    Ensure no two vessels overlap on the same berth and no vessel
    is scheduled before its ETA. Push conflicting starts forward.
    """
    berth_schedule: dict[str, list[tuple[datetime, datetime]]] = {}
    repaired: Chromosome = []

    for vessel_index, berth_id, start_time in chromosome:
        vessel        = vessels[vessel_index]
        start_time    = max(start_time, vessel.eta)          # respect ETA
        duration      = service_times[vessel.name]
        start_time    = _resolve_berth_conflict(berth_id, start_time,
                                                 duration, berth_schedule)
        end_time      = start_time + timedelta(hours=duration)
        berth_schedule.setdefault(berth_id, []).append((start_time, end_time))
        repaired.append((vessel_index, berth_id, start_time))

    return repaired


def _resolve_berth_conflict(berth_id: str, proposed_start: datetime,
                             duration_hours: float,
                             berth_schedule: dict) -> datetime:
    """Push proposed_start forward until it no longer overlaps existing slots."""
    existing_slots = berth_schedule.get(berth_id, [])
    proposed_end   = proposed_start + timedelta(hours=duration_hours)

    for slot_start, slot_end in sorted(existing_slots):
        if proposed_start < slot_end and proposed_end > slot_start:
            proposed_start = slot_end
            proposed_end   = proposed_start + timedelta(hours=duration_hours)

    return proposed_start


# ── Evolution loop ────────────────────────────────────────────────────────────

def evolve_one_generation(population: Population,
                           fitness_fn: Callable[[Chromosome], float],
                           vessels: list[Vessel],
                           eligible_berths: EligibleBerthMap,
                           service_times: ServiceTimeMap) -> Population:
    """Produce the next generation via selection, crossover, mutation, repair."""
    next_gen: Population = []

    while len(next_gen) < len(population):
        parent_a   = tournament_selection(population, fitness_fn)
        parent_b   = tournament_selection(population, fitness_fn)

        if random.random() < GA_CROSSOVER_PROBABILITY:
            child = order_based_crossover(parent_a, parent_b)
        else:
            child = copy.deepcopy(parent_a)

        if random.random() < GA_MUTATION_PROBABILITY:
            child = mutate(child, vessels, eligible_berths)

        child = repair_chromosome(child, vessels, eligible_berths, service_times)
        next_gen.append(child)

    return next_gen


def run_genetic_algorithm(vessels: list[Vessel],
                           eligible_berths: EligibleBerthMap,
                           service_times: ServiceTimeMap,
                           tracker: PortTracker) -> tuple[Chromosome, list[float]]:
    """
    Main GA loop. Returns (best_chromosome, fitness_history).
    Stops after GA_MAX_GENERATIONS or GA_NO_IMPROVE_PATIENCE stagnation.
    """
    greedy_seed  = greedy_dispatch(vessels, eligible_berths, service_times, tracker)
    population   = initialize_population(vessels, eligible_berths,
                                         service_times, tracker, greedy_seed)

    fitness_fn        = lambda c: compute_fitness(c, vessels)
    best_chromosome   = max(population, key=fitness_fn)
    best_fitness      = fitness_fn(best_chromosome)
    fitness_history   = [best_fitness]
    no_improve_count  = 0

    for generation in range(GA_MAX_GENERATIONS):
        population      = evolve_one_generation(population, fitness_fn,
                                                 vessels, eligible_berths, service_times)
        gen_best        = max(population, key=fitness_fn)
        gen_best_fitness= fitness_fn(gen_best)

        if gen_best_fitness > best_fitness:
            best_chromosome  = gen_best
            best_fitness     = gen_best_fitness
            no_improve_count = 0
        else:
            no_improve_count += 1

        fitness_history.append(best_fitness)

        if no_improve_count >= GA_NO_IMPROVE_PATIENCE:
            logger.info("GA converged at generation %d (no improve for %d gen)",
                        generation, GA_NO_IMPROVE_PATIENCE)
            break

    logger.info("GA finished: best fitness=%.6f  total_wait=%.1f h",
                best_fitness,
                compute_total_waiting_hours(best_chromosome, vessels))
    return best_chromosome, fitness_history


# ── Schedule builder ──────────────────────────────────────────────────────────

def chromosome_to_schedule(chromosome: Chromosome,
                            vessels: list[Vessel],
                            berth_matrix: dict[str, Berth],
                            service_times: ServiceTimeMap) -> Schedule:
    """Convert the raw GA output into a rich Schedule domain object."""
    assignments = []
    for vessel_index, berth_id, start_time in chromosome:
        vessel       = vessels[vessel_index]
        berth        = berth_matrix[berth_id]
        duration     = service_times[vessel.name]
        end_time     = start_time + timedelta(hours=duration)
        wait_hours   = max(0.0, (start_time - vessel.eta).total_seconds() / 3600)
        assignments.append(Assignment(
            vessel               = vessel,
            berth                = berth,
            scheduled_start      = start_time,
            predicted_end        = end_time,
            predicted_wait_hours = wait_hours,
        ))
    return Schedule.from_assignments(assignments)
