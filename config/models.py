"""
Domain value objects shared across every module.
These are plain dataclasses — no business logic, no I/O.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Vessel:
    """Immutable snapshot of a vessel at the moment it announces its ETA."""
    name:             str
    vessel_type:      str
    gross_tonnage:    int
    length_m:         float
    draft_m:          float
    cargo_import:     str
    cargo_export:     str
    tonnage_import:   float
    tonnage_export:   float
    tugs_at_berthing: int
    arrival_month:    int
    arrival_dow:      int          # 0 = Monday … 6 = Sunday
    eta:              datetime
    is_priority:      bool = False


@dataclass(frozen=True)
class Berth:
    """Immutable physical and operational specification of a single berth."""
    berth_id:          str
    terminal_name:     str
    quay_length_m:     float
    depth_m:           float
    allowed_cargos:    frozenset[str]
    allowed_types:     frozenset[str]   # vessel types this berth accepts
    operator:          str


@dataclass
class BerthReservation:
    """A single time slot blocked on a berth for one vessel."""
    vessel_name:  str
    berth_id:     str
    start_time:   datetime
    end_time:     datetime

    @property
    def duration_hours(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 3600

    def overlaps_with(self, other: BerthReservation) -> bool:
        return self.start_time < other.end_time and self.end_time > other.start_time


@dataclass
class Assignment:
    """The output of the optimizer for a single vessel."""
    vessel:               Vessel
    berth:                Berth
    scheduled_start:      datetime
    predicted_end:        datetime
    predicted_wait_hours: float

    @property
    def is_direct_berthing(self) -> bool:
        """True when the vessel berths immediately upon arrival."""
        return self.predicted_wait_hours < 0.1


@dataclass
class Schedule:
    """Complete output of one optimizer run — a list of assignments."""
    assignments:          list[Assignment]
    total_wait_hours:     float
    avg_wait_hours:       float
    generated_at:         datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_assignments(cls, assignments: list[Assignment]) -> Schedule:
        total = sum(a.predicted_wait_hours for a in assignments)
        avg   = total / len(assignments) if assignments else 0.0
        return cls(assignments=assignments, total_wait_hours=total, avg_wait_hours=avg)


@dataclass
class PortSnapshot:
    """Instantaneous state of every berth — fed back into the optimizer."""
    timestamp:             datetime
    occupied_berths:       dict[str, BerthReservation]   # berth_id → reservation
    free_berths:           list[str]
    vessel_queue:          list[Vessel]
    queue_length:          int

    @property
    def utilization_rate(self) -> float:
        total = len(self.occupied_berths) + len(self.free_berths)
        return len(self.occupied_berths) / total if total > 0 else 0.0
