# Berth Allocation Agent — Port of Casablanca

A production-grade **Dynamic Discrete Berth Allocation** system that
combines machine learning, a genetic algorithm optimizer, and a
discrete-event simulation engine to minimize vessel anchorage waiting
times at the Port of Casablanca.

---

## Project structure

```
berth_agent/
├── config/
│   ├── constants.py          ← ALL literals live here — no magic values elsewhere
│   └── models.py             ← Immutable domain objects (Vessel, Berth, Schedule …)
│
├── pipeline/
│   ├── data_pipeline.py      ← Layer 1: load → parse → derive → clean → encode
│   └── vessel_builder.py     ← CSV row / AIS message → Vessel domain object
│
├── ml/
│   └── service_time_predictor.py  ← Layer 2A: train, save, load, predict
│
├── constraints/
│   └── constraint_engine.py  ← Layer 2B: filter berths by physical + cargo rules
│
├── optimizer/
│   └── genetic_algorithm.py  ← Layer 3: GA with greedy warm-start + repair
│
├── tracker/
│   └── port_tracker.py       ← Layer 4: berth timelines, assign, release, snapshot
│
├── simulation/
│   ├── scenario_engine.py    ← Layer 5A: 7 SimPy scenarios (S1–S7)
│   └── kpi_reporter.py       ← KPI computation + scenario comparison table
│
├── dashboard/
│   └── port_dashboard.py     ← Layer 5B: Streamlit dashboard (Gantt, alerts, map)
│
├── tests/
│   ├── test_all.py           ← Unit tests: models, pipeline, constraints, tracker, GA
│   └── test_extended.py      ← Unit tests: vessel_builder, KPI reporter, agent wiring
│
├── agent.py                  ← Orchestrator (Facade) — wires all modules together
├── main.py                   ← CLI entry point: train | allocate | simulate | dashboard
└── requirements.txt
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data files
cp casa_history_data.csv  berth_agent/data/
cp port_details.json      berth_agent/data/

# 3. Train the service time predictor
cd berth_agent
python main.py train

# 4. Run a demo allocation
python main.py allocate

# 5. Run all 7 simulation scenarios
python main.py simulate

# 6. Launch the dashboard
python main.py dashboard
# → Open http://localhost:8501 in your browser

# 7. Run all unit tests
python -m pytest tests/ -v
```

Set `BERTH_ENV=production` to use XGBoost instead of the LinearRegression baseline:
```bash
BERTH_ENV=production python main.py train
```

---

## Data flow

```
casa_history_data.csv  ──┐
                         ├─→ DataPipeline ──→ clean DataFrame
port_details.json      ──┘          │
                                    ├─→ BerthConstraintMatrix (dict[str, Berth])
                                    │
                                    ├─→ ML Training → service_predictor.pkl
                                    │
New vessel ETA arrives              │
        │                           ▼
        └──→ VesselBuilder ──→ Vessel
                                    │
                    ┌───────────────┤
                    ▼               ▼
          ConstraintEngine    ServiceTimePredictor
          (eligible berths)   (predicted hours)
                    │               │
                    └───────┬───────┘
                            ▼
                   GeneticAlgorithm
                   (minimize Σ waiting)
                            │
                            ▼
                        Schedule  ──→ PortTracker (register)
                            │
                            ▼
                     Dashboard / KPI Report
```

---

## KPIs

| KPI | Formula |
|---|---|
| Avg waiting time | `mean(berthing_time − arrival_time)` |
| Max waiting time | `max(berthing_time − arrival_time)` |
| Direct berthing rate | `count(wait < 6 min) / total` |
| Berth utilization | `Σ service_hours / (n_berths × 24h)` |
| Schedule accuracy | `count(\|predicted_ETB − actual_ETB\| < 1h) / total` |
| ML RMSE / MAE / R² | From time-based 80/20 train-test split |
