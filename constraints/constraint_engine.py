"""
Layer 2B — Constraint Engine.

Responsibility: given a Vessel and the full berth matrix,
return only the berths that satisfy ALL hard physical and
operational constraints. No optimization happens here.

SOLID note: Open/Closed — new constraint types can be added
as new _check_* functions without touching existing ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.models import Berth, Vessel

logger = logging.getLogger(__name__)


# ── Individual constraint checks ──────────────────────────────────────────────

def _berth_depth_fits_vessel(berth: Berth, vessel: Vessel) -> bool:
    """Hard physical rule: berth water depth must accommodate vessel draft."""
    return berth.depth_m >= vessel.draft_m


def _berth_length_fits_vessel(berth: Berth, vessel: Vessel) -> bool:
    """Hard physical rule: quay must be long enough for vessel."""
    return berth.quay_length_m >= vessel.length_m


def _berth_accepts_cargo(berth: Berth, vessel: Vessel) -> bool:
    """
    Operational rule: berth must handle at least one of the vessel's cargos.
    Empty allowed_cargos means the berth has no restriction (general cargo).
    """
    if not berth.allowed_cargos:
        return True
    vessel_cargos = {vessel.cargo_import, vessel.cargo_export} - {"", "nan", "NaN"}
    if not vessel_cargos:
        return True
    return bool(berth.allowed_cargos & vessel_cargos)


def _berth_accepts_vessel_type(berth: Berth, vessel: Vessel) -> bool:
    """
    Operational rule: berth must accept the vessel type.
    Empty allowed_types means the berth is type-agnostic.
    """
    if not berth.allowed_types:
        return True
    return vessel.vessel_type in berth.allowed_types


# ── Constraint evaluation record ──────────────────────────────────────────────

@dataclass(frozen=True)
class ConstraintViolation:
    """Explains why a specific berth was rejected for a vessel."""
    berth_id: str
    reason:   str


@dataclass
class EligibilityReport:
    """Full eligibility result for one vessel across all berths."""
    vessel_name:     str
    eligible_berths: list[Berth]
    violations:      list[ConstraintViolation]

    @property
    def has_eligible_berths(self) -> bool:
        return len(self.eligible_berths) > 0

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_berths)


# ── Main constraint engine ────────────────────────────────────────────────────

class ConstraintEngine:
    """
    Filters the global berth matrix to only eligible berths for a vessel.
    One responsibility: constraint checking. No scheduling, no optimization.
    """

    def __init__(self, berth_matrix: dict[str, Berth]) -> None:
        self._berth_matrix = berth_matrix

    def find_eligible_berths(self, vessel: Vessel) -> EligibilityReport:
        """
        Evaluate all hard constraints and return the eligibility report.
        Logs a warning if no berths qualify (port capacity issue).
        """
        eligible:   list[Berth]               = []
        violations: list[ConstraintViolation] = []

        for berth in self._berth_matrix.values():
            violation = self._evaluate_berth_constraints(berth, vessel)
            if violation is None:
                eligible.append(berth)
            else:
                violations.append(violation)

        report = EligibilityReport(vessel.name, eligible, violations)
        self._log_eligibility_result(report)
        return report

    def _evaluate_berth_constraints(self, berth: Berth,
                                    vessel: Vessel) -> ConstraintViolation | None:
        """
        Run all constraint checks on a single berth.
        Return None if the berth is eligible, else return the first violation.
        """
        if not _berth_depth_fits_vessel(berth, vessel):
            return ConstraintViolation(berth.berth_id,
                f"Depth {berth.depth_m}m < vessel draft {vessel.draft_m}m")
        if not _berth_length_fits_vessel(berth, vessel):
            return ConstraintViolation(berth.berth_id,
                f"Length {berth.quay_length_m}m < vessel length {vessel.length_m}m")
        if not _berth_accepts_cargo(berth, vessel):
            return ConstraintViolation(berth.berth_id,
                f"Cargo mismatch: vessel={vessel.cargo_import}/{vessel.cargo_export}")
        if not _berth_accepts_vessel_type(berth, vessel):
            return ConstraintViolation(berth.berth_id,
                f"Type mismatch: vessel type '{vessel.vessel_type}' not accepted")
        return None

    def _log_eligibility_result(self, report: EligibilityReport) -> None:
        if report.has_eligible_berths:
            logger.debug("Vessel %s → %d eligible berths",
                         report.vessel_name, report.eligible_count)
        else:
            logger.warning("Vessel %s → NO eligible berths found! "
                           "Check draft/length/cargo constraints.", report.vessel_name)

    def get_eligible_berth_ids(self, vessel: Vessel) -> list[str]:
        """Convenience: return just the berth ID strings for a vessel."""
        report = self.find_eligible_berths(vessel)
        return [b.berth_id for b in report.eligible_berths]
