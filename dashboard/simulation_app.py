"""
dashboard/simulation_app.py
════════════════════════════
Interactive Port of Casablanca — Berth Allocation Simulation

Run:  streamlit run dashboard/simulation_app.py

Panels:
  ① Setup        — configure initial berths in service + waiting queue
  ② Port View    — live berth map + queue table + allocation results
  ③ Time Control — advance clock, add vessels, re-run allocation
  ④ KPIs         — cumulative metrics + history charts
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.constants import CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL
from config.models import Berth, Vessel
from dashboard.simulation_state import (
    ActiveBerthSlot, AllocationSnapshot, SimulationState,
    TIME_STEP_OPTIONS_HOURS, create_initial_state,
)
from dashboard.vessel_generator import (
    VESSEL_PROFILES, VESSEL_TYPE_LABELS,
    cargo_options_for_type, generate_random_vessel,
    generate_vessel_fleet, vessel_type_options,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Casa Port — Live Simulation",
    page_icon  = "⚓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }

.port-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #102840 100%);
    border-bottom: 2px solid #00d4ff22;
    padding: 1.2rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; gap: 1rem;
}
.port-header h1 {
    color: #00d4ff; font-size: 1.4rem; margin: 0;
    text-shadow: 0 0 20px #00d4ff44;
}
.port-header .clock {
    margin-left: auto; color: #7fb3cc;
    font-family: 'Space Mono', monospace; font-size: 0.9rem;
}

.metric-card {
    background: #0d1f2d;
    border: 1px solid #1a3a52;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #00d4ff;
}
.metric-card .label {
    font-size: 0.75rem; color: #5a8aa0;
    text-transform: uppercase; letter-spacing: 1px;
}

.berth-card {
    border-radius: 8px; padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem; border-left: 3px solid;
    font-size: 0.82rem;
}
.berth-occupied {
    background: #0d2137; border-color: #f97316;
    color: #cbd5e1;
}
.berth-free {
    background: #071a0f; border-color: #22c55e;
    color: #86efac;
}
.berth-name {
    font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 0.85rem;
}
.vessel-tag {
    background: #f9731622; color: #fb923c;
    border-radius: 4px; padding: 1px 6px;
    font-size: 0.75rem; font-family: 'Space Mono', monospace;
}
.queue-tag {
    background: #3b82f622; color: #93c5fd;
    border-radius: 4px; padding: 1px 6px;
    font-size: 0.75rem;
}
.priority-tag {
    background: #dc262622; color: #f87171;
    border-radius: 4px; padding: 1px 6px;
    font-size: 0.75rem; font-weight: 600;
}

.step-banner {
    background: #0d1f2d;
    border: 1px solid #00d4ff33;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-family: 'Space Mono', monospace;
    color: #00d4ff; font-size: 0.85rem;
    margin-bottom: 1rem;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem; color: #5a8aa0;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #1a3a52;
    padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem 0;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00d4ff, #0284c7) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state bootstrap ────────────────────────────────────────────────────

@st.cache_resource
def load_berth_matrix() -> dict[str, Berth]:
    """Load and cache the berth matrix — runs once per session."""
    from pipeline.data_pipeline import run_full_pipeline
    _, bm = run_full_pipeline(CSV_PATH, JSON_PATH)
    return bm


@st.cache_resource
def load_agent():
    """Load and cache the allocation agent — runs once per session."""
    if not SERVICE_TIME_MODEL.exists():
        return None
    from agent import build_agent_from_config
    return build_agent_from_config(CSV_PATH, JSON_PATH, SERVICE_TIME_MODEL)


def init_session() -> None:
    """Initialise all session state keys if not already present."""
    if "page" not in st.session_state:
        st.session_state.page = "setup"
    if "sim_state" not in st.session_state:
        st.session_state.sim_state = None
    if "setup_active" not in st.session_state:
        st.session_state.setup_active = []          # list[dict] editable rows
    if "setup_queue" not in st.session_state:
        st.session_state.setup_queue = []           # list[dict] editable rows
    if "add_vessel_open" not in st.session_state:
        st.session_state.add_vessel_open = False


init_session()


# ── Helpers ───────────────────────────────────────────────────────────────────

def vessel_to_dict(v: Vessel) -> dict:
    return {
        "name":            v.name,
        "vessel_type":     v.vessel_type,
        "length_m":        v.length_m,
        "draft_m":         v.draft_m,
        "gross_tonnage":   v.gross_tonnage,
        "cargo_import":    v.cargo_import,
        "cargo_export":    v.cargo_export,
        "tonnage_import":  v.tonnage_import,
        "tonnage_export":  v.tonnage_export,
        "tugs_at_berthing":v.tugs_at_berthing,
        "eta":             v.eta,
        "is_priority":     v.is_priority,
    }


def dict_to_vessel(d: dict) -> Vessel:
    eta = d["eta"]
    if isinstance(eta, str):
        eta = datetime.fromisoformat(eta)
    return Vessel(
        name             = d["name"],
        vessel_type      = d["vessel_type"],
        gross_tonnage    = int(d["gross_tonnage"]),
        length_m         = float(d["length_m"]),
        draft_m          = float(d["draft_m"]),
        cargo_import     = d["cargo_import"],
        cargo_export     = d.get("cargo_export", ""),
        tonnage_import   = float(d["tonnage_import"]),
        tonnage_export   = float(d.get("tonnage_export", 0)),
        tugs_at_berthing = int(d["tugs_at_berthing"]),
        arrival_month    = eta.month,
        arrival_dow      = eta.weekday(),
        eta              = eta,
        is_priority      = bool(d.get("is_priority", False)),
    )


def pick_random_berth_for_vessel(vessel: Vessel,
                                  berth_matrix: dict[str, Berth],
                                  exclude: set[str]) -> Berth | None:
    """Return a random eligible berth not in `exclude` (already occupied)."""
    from constraints.constraint_engine import ConstraintEngine
    engine    = ConstraintEngine(berth_matrix)
    eligible  = engine.find_eligible_berths(vessel).eligible_berths
    available = [b for b in eligible if b.berth_id not in exclude]
    return random.choice(available) if available else None


def run_allocation(state: SimulationState) -> None:
    """Run the GA optimizer on the current waiting queue and apply results."""
    agent = load_agent()
    if agent is None:
        st.error("⚠️ Model not trained. Run `python main.py train` first.")
        return

    if not state.waiting_queue:
        st.info("Queue is empty — nothing to allocate.")
        return

    with st.spinner("Running GA optimizer…"):
        # Refresh tracker with current active slots
        from tracker.port_tracker import PortTracker
        tracker = PortTracker(state.berth_matrix)
        for slot in state.active_slots:
            try:
                tracker.assign_vessel_to_berth(
                    slot.vessel, slot.berth.berth_id,
                    slot.arrived_at, slot.departs_at)
            except Exception:
                pass   # skip if conflict in initial seeded state

        # Rebuild constraint engine and predictor
        from constraints.constraint_engine import ConstraintEngine
        from ml.service_time_predictor import ServiceTimePredictor
        engine    = ConstraintEngine(state.berth_matrix)
        predictor = ServiceTimePredictor(SERVICE_TIME_MODEL)

        # Predict service times
        service_times = {
            v.name: predictor.predict_service_hours(v)
            for v in state.waiting_queue
        }

        # Filter eligible vessels
        eligible = {
            v.name: engine.find_eligible_berths(v).eligible_berths
            for v in state.waiting_queue
        }
        feasible = [v for v in state.waiting_queue if eligible.get(v.name)]

        if not feasible:
            st.warning("No vessels in queue have eligible berths.")
            return

        from optimizer.genetic_algorithm import (
            run_genetic_algorithm, chromosome_to_schedule,
        )
        best_chrom, _ = run_genetic_algorithm(
            vessels         = feasible,
            eligible_berths = eligible,
            service_times   = service_times,
            tracker         = tracker,
        )
        schedule = chromosome_to_schedule(
            best_chrom, feasible, state.berth_matrix, service_times)

    state.apply_schedule(schedule)
    st.success(f"✅ Allocated {len(schedule.assignments)} vessels — "
               f"avg wait {schedule.avg_wait_hours:.1f} h")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ══════════════════════════════════════════════════════════════════════════════

def render_setup_page() -> None:
    berth_matrix = load_berth_matrix()

    st.markdown("## ⚙️ Simulation Setup")
    st.markdown("Configure the initial port state before starting the simulation.")

    col_l, col_r = st.columns(2, gap="large")

    # ── LEFT: vessels currently in berths ─────────────────────────────────────
    with col_l:
        st.markdown('<p class="section-title">🟠 Vessels currently in service</p>',
                    unsafe_allow_html=True)

        n_active = st.number_input("Number of vessels already berthed",
                                    min_value=0, max_value=len(berth_matrix),
                                    value=min(5, len(berth_matrix)), step=1,
                                    key="n_active")

        if st.button("🎲 Generate random active vessels", key="gen_active"):
            st.session_state.setup_active = []
            occupied_berths: set[str] = set()
            now = datetime.utcnow()

            profiles_cycle = VESSEL_PROFILES * 10
            for i in range(int(n_active)):
                profile = profiles_cycle[i % len(VESSEL_PROFILES)]
                v       = generate_random_vessel(
                    eta=now - timedelta(hours=random.uniform(2, 20)))
                berth   = pick_random_berth_for_vessel(v, berth_matrix, occupied_berths)
                if berth is None:
                    continue
                occupied_berths.add(berth.berth_id)
                service_hrs = random.uniform(12, 72)
                st.session_state.setup_active.append({
                    "vessel":      vessel_to_dict(v),
                    "berth_id":    berth.berth_id,
                    "arrived_at":  now - timedelta(hours=random.uniform(2, 20)),
                    "departs_at":  now + timedelta(hours=service_hrs),
                })

        if st.session_state.setup_active:
            st.markdown(f"**{len(st.session_state.setup_active)} vessels in service:**")
            for i, row in enumerate(st.session_state.setup_active):
                with st.expander(
                    f"🚢 {row['vessel']['name']}  →  Berth {row['berth_id']}", expanded=False
                ):
                    _render_active_vessel_editor(i, row, berth_matrix)

    # ── RIGHT: waiting queue ──────────────────────────────────────────────────
    with col_r:
        st.markdown('<p class="section-title">🔵 Vessels waiting for allocation</p>',
                    unsafe_allow_html=True)

        n_queue = st.number_input("Number of vessels in waiting queue",
                                   min_value=0, max_value=30,
                                   value=6, step=1, key="n_queue")

        if st.button("🎲 Generate random waiting vessels", key="gen_queue"):
            now    = datetime.utcnow()
            fleet  = generate_vessel_fleet(int(n_queue), now, spread_hours=12)
            st.session_state.setup_queue = [vessel_to_dict(v) for v in fleet]

        if st.session_state.setup_queue:
            st.markdown(f"**{len(st.session_state.setup_queue)} vessels in queue:**")
            for i, row in enumerate(st.session_state.setup_queue):
                with st.expander(f"⏳ {row['name']}  — ETA {row['eta'].strftime('%H:%M') if isinstance(row['eta'], datetime) else row['eta']}", expanded=False):
                    _render_queue_vessel_editor(i, row)

    st.divider()

    # ── Start time config ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🕐 Simulation clock</p>',
                unsafe_allow_html=True)
    start_date = st.date_input("Start date", value=datetime.utcnow().date())
    start_hour = st.slider("Start hour", 0, 23,
                            value=datetime.utcnow().hour, key="start_hour")
    sim_start  = datetime(start_date.year, start_date.month, start_date.day,
                          start_hour, 0, 0)

    # ── Launch ────────────────────────────────────────────────────────────────
    if st.button("🚀 Launch Simulation", type="primary", use_container_width=True):
        active_slots = _build_active_slots_from_setup(berth_matrix)
        queue        = [dict_to_vessel(d) for d in st.session_state.setup_queue]
        state        = create_initial_state(berth_matrix, active_slots, queue, sim_start)
        st.session_state.sim_state = state
        st.session_state.page      = "simulation"
        st.rerun()


def _render_active_vessel_editor(idx: int, row: dict, berth_matrix: dict) -> None:
    """Inline editor for a vessel already in service."""
    v = row["vessel"]
    c1, c2 = st.columns(2)
    with c1:
        v["name"]         = st.text_input("Name", v["name"], key=f"act_name_{idx}")
        v["length_m"]     = st.number_input("Length (m)", 50.0, 400.0, v["length_m"], 1.0, key=f"act_len_{idx}")
        v["draft_m"]      = st.number_input("Draft (m)", 4.0, 14.0, v["draft_m"], 0.1, key=f"act_dft_{idx}")
        v["gross_tonnage"]= st.number_input("Gross tonnage", 1000, 200000, v["gross_tonnage"], 500, key=f"act_gt_{idx}")
    with c2:
        v["vessel_type"]  = st.selectbox("Type", vessel_type_options(),
                                          index=vessel_type_options().index(v["vessel_type"])
                                          if v["vessel_type"] in vessel_type_options() else 0,
                                          key=f"act_type_{idx}")
        row["berth_id"]   = st.selectbox("Berth", sorted(berth_matrix.keys()),
                                          index=sorted(berth_matrix.keys()).index(row["berth_id"])
                                          if row["berth_id"] in berth_matrix else 0,
                                          key=f"act_berth_{idx}")
        departs = row["departs_at"]
        if isinstance(departs, datetime):
            hours_left = st.number_input("Hours remaining", 0.0, 200.0,
                                          round((departs - datetime.utcnow()).total_seconds() / 3600, 1),
                                          0.5, key=f"act_rem_{idx}")
            row["departs_at"] = datetime.utcnow() + timedelta(hours=hours_left)
    st.session_state.setup_active[idx] = row


def _render_queue_vessel_editor(idx: int, row: dict) -> None:
    """Inline editor for a vessel in the waiting queue."""
    c1, c2 = st.columns(2)
    with c1:
        row["name"]         = st.text_input("Name", row["name"], key=f"q_name_{idx}")
        row["length_m"]     = st.number_input("Length (m)", 50.0, 400.0, float(row["length_m"]), 1.0, key=f"q_len_{idx}")
        row["draft_m"]      = st.number_input("Draft (m)", 4.0, 14.0, float(row["draft_m"]), 0.1, key=f"q_dft_{idx}")
        row["gross_tonnage"]= st.number_input("Gross tonnage", 1000, 200000, int(row["gross_tonnage"]), 500, key=f"q_gt_{idx}")
    with c2:
        row["vessel_type"]  = st.selectbox("Type", vessel_type_options(),
                                            index=vessel_type_options().index(row["vessel_type"])
                                            if row["vessel_type"] in vessel_type_options() else 0,
                                            key=f"q_type_{idx}")
        row["cargo_import"] = st.text_input("Cargo import", row["cargo_import"], key=f"q_cimp_{idx}")
        row["tonnage_import"]= st.number_input("Tonnage import", 0.0, 200000.0, float(row["tonnage_import"]), 100.0, key=f"q_timp_{idx}")
        row["is_priority"]  = st.checkbox("Priority vessel", bool(row.get("is_priority", False)), key=f"q_pri_{idx}")
        eta = row["eta"]
        if isinstance(eta, datetime):
            hours_from_now = st.number_input("ETA from now (h)", -24.0, 72.0,
                                              round((eta - datetime.utcnow()).total_seconds() / 3600, 1),
                                              0.5, key=f"q_eta_{idx}")
            row["eta"] = datetime.utcnow() + timedelta(hours=hours_from_now)
    st.session_state.setup_queue[idx] = row


def _build_active_slots_from_setup(berth_matrix: dict[str, Berth]) -> list[ActiveBerthSlot]:
    """Convert setup_active dicts into ActiveBerthSlot objects."""
    slots = []
    for row in st.session_state.setup_active:
        try:
            vessel  = dict_to_vessel(row["vessel"])
            berth   = berth_matrix.get(row["berth_id"])
            if berth is None:
                continue
            slots.append(ActiveBerthSlot(
                vessel     = vessel,
                berth      = berth,
                arrived_at = row.get("arrived_at", datetime.utcnow() - timedelta(hours=6)),
                departs_at = row["departs_at"],
            ))
        except Exception:
            pass
    return slots


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def render_simulation_page() -> None:
    state: SimulationState = st.session_state.sim_state
    SimulationState._current_time_ref = state.sim_time

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="port-header">
        <span style="font-size:1.8rem;">⚓</span>
        <h1>Port of Casablanca — Live Simulation</h1>
        <div class="clock">
            Step {state.step_number} &nbsp;|&nbsp;
            🕐 {state.sim_time.strftime("%Y-%m-%d  %H:%M")} UTC
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    _kpi_card(k1, str(state.occupied_berth_count),    "Berths occupied")
    _kpi_card(k2, str(state.free_berth_count),         "Berths free")
    _kpi_card(k3, str(len(state.waiting_queue)),       "In queue")
    _kpi_card(k4, str(state.total_vessels_served),     "Total served")
    _kpi_card(k5, f"{state.avg_cumulative_wait:.1f}h", "Avg wait (all)")

    st.markdown("---")

    # ── Main two-column layout ─────────────────────────────────────────────────
    col_port, col_ctrl = st.columns([3, 2], gap="large")

    with col_port:
        _render_berth_map(state)
        st.markdown("---")
        _render_waiting_queue(state)

    with col_ctrl:
        _render_time_control(state)
        st.markdown("---")
        _render_add_vessel_panel(state)
        st.markdown("---")
        _render_last_allocation_result(state)

    # ── KPI charts at bottom ──────────────────────────────────────────────────
    if state.kpi_history:
        st.markdown("---")
        _render_kpi_charts(state)


def _kpi_card(col, value: str, label: str) -> None:
    col.markdown(f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_berth_map(state: SimulationState) -> None:
    st.markdown('<p class="section-title">🗺️ Berth occupancy map</p>',
                unsafe_allow_html=True)

    occupied_map = {s.berth.berth_id: s for s in state.active_slots}

    # Group berths by terminal
    terminal_groups: dict[str, list[Berth]] = {}
    for berth in state.berth_matrix.values():
        terminal_groups.setdefault(berth.terminal_name, []).append(berth)

    for terminal_name, berths in sorted(terminal_groups.items()):
        with st.expander(f"🏭 {terminal_name}  ({len(berths)} berths)", expanded=True):
            cols = st.columns(min(4, len(berths)))
            for i, berth in enumerate(sorted(berths, key=lambda b: b.berth_id)):
                col = cols[i % len(cols)]
                slot = occupied_map.get(berth.berth_id)
                if slot:
                    remaining = slot.remaining_hours
                    progress  = slot.progress_pct
                    col.markdown(f"""
                    <div class="berth-card berth-occupied">
                        <div class="berth-name">BERTH {berth.berth_id}</div>
                        <div><span class="vessel-tag">{slot.vessel.name}</span></div>
                        <div style="color:#94a3b8;margin-top:3px;font-size:0.75rem;">
                            ⏱ {remaining:.1f}h remaining
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    col.progress(int(progress))
                else:
                    col.markdown(f"""
                    <div class="berth-card berth-free">
                        <div class="berth-name">BERTH {berth.berth_id}</div>
                        <div style="font-size:0.75rem;">✅ FREE</div>
                        <div style="font-size:0.7rem;color:#4ade80;margin-top:2px;">
                            depth {berth.depth_m}m · {berth.quay_length_m:.0f}m
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


def _render_waiting_queue(state: SimulationState) -> None:
    st.markdown('<p class="section-title">⏳ Waiting queue</p>',
                unsafe_allow_html=True)

    if not state.waiting_queue:
        st.info("Queue is empty — all vessels have been allocated.")
        return

    rows = []
    for rank, v in enumerate(state.waiting_queue, 1):
        wait_since = max(0.0, (state.sim_time - v.eta).total_seconds() / 3600)
        rows.append({
            "Rank":        rank,
            "Vessel":      v.name,
            "Type":        VESSEL_TYPE_LABELS.get(v.vessel_type, v.vessel_type),
            "Draft (m)":   v.draft_m,
            "Length (m)":  v.length_m,
            "Cargo":       v.cargo_import or v.cargo_export,
            "ETA":         v.eta.strftime("%Y-%m-%d %H:%M"),
            "Waiting (h)": round(wait_since, 1),
            "Priority":    "🔴 YES" if v.is_priority else "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
                  column_config={
                      "Waiting (h)": st.column_config.NumberColumn(format="%.1f ⏱"),
                      "Priority":    st.column_config.TextColumn(),
                  })


def _render_time_control(state: SimulationState) -> None:
    st.markdown('<p class="section-title">⏩ Time control</p>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="step-banner">
        Current time: {state.sim_time.strftime("%Y-%m-%d  %H:%M")} UTC
        &nbsp;·&nbsp; Step #{state.step_number}
    </div>
    """, unsafe_allow_html=True)

    step_label = st.radio("Advance clock by", list(TIME_STEP_OPTIONS_HOURS.keys()),
                           horizontal=True, key="time_step_choice")
    step_hours = TIME_STEP_OPTIONS_HOURS[step_label]

    if st.button(f"⏩ Advance {step_label} then re-allocate",
                  type="primary", use_container_width=True):
        completed = state.advance_clock(step_hours)
        if completed:
            st.toast(f"✅ {len(completed)} vessel(s) departed", icon="🚢")
        run_allocation(state)

    if st.button("🔄 Run allocation NOW (no clock advance)",
                  use_container_width=True):
        run_allocation(state)

    if st.button("↩️ Back to setup", use_container_width=True):
        st.session_state.page = "setup"
        st.session_state.sim_state = None
        st.rerun()


def _render_add_vessel_panel(state: SimulationState) -> None:
    st.markdown('<p class="section-title">➕ Add vessel to queue</p>',
                unsafe_allow_html=True)

    with st.expander("Add a new vessel", expanded=False):
        tab_random, tab_manual = st.tabs(["🎲 Random vessel", "✏️ Manual entry"])

        with tab_random:
            profile_names = [
                f"{VESSEL_TYPE_LABELS.get(p['vessel_type'], p['vessel_type'])} — "
                f"{p['cargo_import'] or p['cargo_export']}"
                for p in VESSEL_PROFILES
            ]
            profile_idx = st.selectbox("Vessel profile", range(len(VESSEL_PROFILES)),
                                        format_func=lambda i: profile_names[i],
                                        key="rand_profile")
            eta_offset  = st.number_input("ETA from now (hours)", 0.0, 72.0,
                                           2.0, 0.5, key="rand_eta_off")
            is_priority = st.checkbox("Mark as priority", key="rand_prio")
            if st.button("Add random vessel", key="add_rand"):
                eta = state.sim_time + timedelta(hours=eta_offset)
                v   = generate_random_vessel(eta=eta, profile_index=int(profile_idx))
                if is_priority:
                    v = Vessel(**{**v.__dict__, "is_priority": True})
                state.add_vessel_to_queue(v)
                st.success(f"Added {v.name} to queue (ETA {eta.strftime('%H:%M')})")
                st.rerun()

        with tab_manual:
            name         = st.text_input("Vessel name", "MY_VESSEL_01", key="man_name")
            vtype        = st.selectbox("Type", vessel_type_options(), key="man_type")
            cargo_imp    = st.text_input("Cargo import", "Containers", key="man_cimp")
            cargo_exp    = st.text_input("Cargo export", "", key="man_cexp")
            length       = st.number_input("Length (m)", 50.0, 400.0, 180.0, 1.0, key="man_len")
            draft        = st.number_input("Draft (m)", 4.0, 14.0, 9.5, 0.1, key="man_dft")
            tonnage      = st.number_input("Gross tonnage", 1000, 200000, 30000, 500, key="man_gt")
            t_import     = st.number_input("Tonnage import", 0.0, 200000.0, 5000.0, 100.0, key="man_timp")
            eta_off_m    = st.number_input("ETA from now (h)", 0.0, 72.0, 3.0, 0.5, key="man_eta")
            is_prio_m    = st.checkbox("Priority", key="man_prio")

            if st.button("Add this vessel", key="add_man"):
                eta = state.sim_time + timedelta(hours=eta_off_m)
                v   = Vessel(
                    name=name, vessel_type=vtype,
                    gross_tonnage=int(tonnage), length_m=length, draft_m=draft,
                    cargo_import=cargo_imp, cargo_export=cargo_exp,
                    tonnage_import=t_import, tonnage_export=0.0,
                    tugs_at_berthing=2, arrival_month=eta.month,
                    arrival_dow=eta.weekday(), eta=eta, is_priority=is_prio_m,
                )
                state.add_vessel_to_queue(v)
                st.success(f"Added {name} to queue")
                st.rerun()


def _render_last_allocation_result(state: SimulationState) -> None:
    if not state.allocation_history:
        st.info("No allocation run yet — advance the clock or press 'Run allocation'.")
        return

    st.markdown('<p class="section-title">📋 Last allocation result</p>',
                unsafe_allow_html=True)

    last: AllocationSnapshot = state.allocation_history[-1]
    sched = last.schedule

    c1, c2, c3 = st.columns(3)
    c1.metric("Vessels assigned", len(sched.assignments))
    c2.metric("Avg wait",          f"{sched.avg_wait_hours:.1f} h")
    c3.metric("Queue remaining",   last.queue_after)

    if sched.assignments:
        rows = []
        for a in sorted(sched.assignments, key=lambda x: x.scheduled_start):
            rows.append({
                "Vessel":   a.vessel.name,
                "Berth":    a.berth.berth_id,
                "Terminal": a.berth.terminal_name[:20],
                "ETB":      a.scheduled_start.strftime("%m-%d %H:%M"),
                "ETD":      a.predicted_end.strftime("%m-%d %H:%M"),
                "Wait (h)": round(a.predicted_wait_hours, 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_kpi_charts(state: SimulationState) -> None:
    st.markdown('<p class="section-title">📊 KPI evolution over time</p>',
                unsafe_allow_html=True)

    df = pd.DataFrame(state.kpi_history)

    tab1, tab2, tab3 = st.tabs(["Waiting time", "Queue & Berths", "Throughput"])

    with tab1:
        import plotly.express as px
        fig = px.bar(df, x="sim_time", y=["avg_wait_h", "total_wait_h"],
                      barmode="group",
                      labels={"value": "Hours", "variable": "Metric",
                               "sim_time": "Simulation time"},
                      title="Waiting time per allocation step",
                      color_discrete_sequence=["#00d4ff", "#f97316"])
        fig.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628",
                           font_color="#94a3b8", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.line(df, x="sim_time",
                        y=["queue_length", "berths_occupied"],
                        markers=True,
                        labels={"value": "Count", "variable": "Series",
                                 "sim_time": "Simulation time"},
                        title="Queue length & berth occupancy over time",
                        color_discrete_sequence=["#3b82f6", "#f97316"])
        fig2.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628",
                            font_color="#94a3b8", legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.bar(df, x="sim_time", y="vessels_served",
                       labels={"vessels_served": "Vessels allocated",
                                "sim_time": "Simulation time"},
                       title="Vessels allocated per step",
                       color="utilization_pct",
                       color_continuous_scale="Blues")
        fig3.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628",
                            font_color="#94a3b8")
        st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    with st.expander("Raw KPI table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    agent_ready = load_agent() is not None
    if not agent_ready:
        st.warning(
            "⚠️ Trained model not found. "
            "Run `python main.py train` from the `berth_agent/` directory first, "
            "then refresh this page."
        )

    page = st.session_state.get("page", "setup")

    if page == "setup":
        render_setup_page()
    elif page == "simulation":
        if st.session_state.sim_state is None:
            st.session_state.page = "setup"
            st.rerun()
        render_simulation_page()


if __name__ == "__main__":
    main()
