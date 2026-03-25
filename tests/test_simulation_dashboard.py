"""
Test script for port_simulation.py.
Tests initialization, UI elements, and basic functionality.
"""

import sys
import time
from datetime import datetime, timedelta, timedelta

import pytest

sys.path.insert(0, '.')


def test_port_simulation():
    from dashboard.port_simulation import (
        initialize_berthed_vessels,
        initialize_waiting_queue,
        initialize_port_state,
        random_vessel,
        dict_to_vessel,
        vessel_to_dict,
        advance_time,
    )
    from pipeline.data_pipeline import run_full_pipeline
    from config.constants import CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL
    from config.models import Vessel
    from agent import build_agent_from_config
    from simulation.scenario_engine import _group_berths_by_terminal
    
    print("=" * 60)
    print("Testing Port Simulation Dashboard")
    print("=" * 60)
    
    # 1. Test data loading
    print("\n[TEST 1] Loading port data...")
    df, berth_matrix = run_full_pipeline(CSV_PATH, JSON_PATH)
    assert len(berth_matrix) == 49, f"Expected 49 berths, got {len(berth_matrix)}"
    print(f"✓ Loaded {len(berth_matrix)} berths, {len(df)} historical records")
    
    # 2. Test random vessel generation
    print("\n[TEST 2] Testing random vessel generation...")
    base_time = datetime(2024, 6, 1, 6, 0)
    v = random_vessel("TEST_VESSEL", base_time)
    assert v.name == "TEST_VESSEL"
    assert v.vessel_type in ["PORTE_CONTENEUR", "VRAQUIER", "ROULIER", "CARGO_GENERAL", "CABOTIER"]
    assert v.gross_tonnage > 0
    assert v.length_m > 0
    assert v.draft_m > 0
    print(f"✓ Generated: {v.name}, type={v.vessel_type}, gt={v.gross_tonnage}, len={v.length_m}m, draft={v.draft_m}m")
    
    # 3. Test vessel conversion
    print("\n[TEST 3] Testing vessel dict conversion...")
    d = vessel_to_dict(v)
    v2 = dict_to_vessel(d)
    assert v2.name == v.name
    assert v2.vessel_type == v.vessel_type
    assert v2.gross_tonnage == v.gross_tonnage
    print("✓ Vessel ↔ Dict conversion works")
    
    # 4. Test initialize_berthed_vessels
    print("\n[TEST 4] Testing berthed vessels initialization...")
    berthed = initialize_berthed_vessels(berth_matrix, 5, base_time, 36.0)
    assert len(berthed) == 5, f"Expected 5, got {len(berthed)}"
    for vessel, berth_id, start, end in berthed:
        assert berth_id in berth_matrix
        assert start < end
    print(f"✓ Created {len(berthed)} berthed vessels")
    
    # 5. Test initialize_waiting_queue
    print("\n[TEST 5] Testing waiting queue initialization...")
    waiting = initialize_waiting_queue(4, base_time)
    assert len(waiting) == 4
    for v in waiting:
        assert v.eta >= base_time
        assert v.eta <= base_time + timedelta(hours=24)
    print(f"✓ Created {len(waiting)} waiting vessels")
    
    # 6. Test initialize_port_state
    print("\n[TEST 6] Testing port state initialization...")
    from agent import build_agent_from_config
    agent = build_agent_from_config(CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL)
    tracker, waiting = initialize_port_state(berth_matrix, 3, 2, base_time, agent)
    occupied = tracker.get_occupied_berths(base_time)
    assert len(occupied) == 3, f"Expected 3 occupied, got {len(occupied)}"
    assert len(waiting) == 2
    print(f"✓ Port state: {len(occupied)} berthed, {len(waiting)} waiting")
    
    # 7. Test allocation
    print("\n[TEST 7] Testing allocation...")
    schedule = agent.allocate(waiting)
    assert schedule is not None
    assert len(schedule.assignments) <= len(waiting)
    print(f"✓ Allocation: {len(schedule.assignments)} vessels assigned")
    print(f"  Total wait: {schedule.total_wait_hours:.1f}h, Avg: {schedule.avg_wait_hours:.1f}h")
    
    # 8. Test time advancement
    print("\n[TEST 8] Testing time advancement...")
    from dashboard.port_simulation import advance_time
    new_time = advance_time(base_time, 48, tracker, berth_matrix)
    assert new_time == base_time + timedelta(hours=48)
    # Some berths should be cleared after 48h (service was 36h)
    print(f"✓ Time advanced to {new_time}")
    
    # 9. Test KPI computation
    print("\n[TEST 9] Testing KPI computation...")
    total_service = sum((a.predicted_end - a.scheduled_start).total_seconds() / 3600 
                        for a in schedule.assignments)
    occupied_now = len(tracker.get_occupied_berths(base_time))
    utilization = occupied_now / len(berth_matrix)
    print(f"  Total service: {total_service:.1f}h, Utilization: {utilization:.1%}")
    print("✓ KPIs computed")
    
    # 10. Test all terminals have berths
    print("\n[TEST 10] Testing terminal coverage...")
    from simulation.scenario_engine import _group_berths_by_terminal
    terminal_berths = _group_berths_by_terminal(berth_matrix)
    assert len(terminal_berths) >= 6, f"Expected 6+ terminals, got {len(terminal_berths)}"
    print(f"✓ {len(terminal_berths)} terminals with berths:")
    for term, berths in terminal_berths.items():
        print(f"    - {term}: {len(berths)} berths")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)

if __name__ == "__main__":
    from datetime import timedelta
    test_port_simulation()