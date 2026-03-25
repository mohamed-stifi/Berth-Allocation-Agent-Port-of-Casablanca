## Complete Berth Allocation Agent — System Design

### 1. Problem Formulation

Your sketch defines a **Dynamic Discrete Berth Allocation Problem (D-DBAP)** with the following formal structure:

**Sets:**
- **V** = set of vessels in the queue, each vessel `vᵢ` characterized by `(ETAᵢ, Lᵢ, Dᵢ, Typeᵢ, Cargoᵢ, Tonnageᵢ)`
- **B** = set of discrete berths from the JSON (e.g., berth 20, 21, 30…83), each berth `bⱼ` characterized by `(LengthⱼDepthⱼ, CargoTypesⱼ, Terminalⱼ)`

**Decision variables:**
- `xᵢⱼ` = 1 if vessel `vᵢ` is assigned to berth `bⱼ`, else 0
- `sᵢ` = start time (berthing time) of vessel `vᵢ` at its assigned berth

**Objective — minimize total anchorage waiting:**
```
minimize Σᵢ (sᵢ − ETAᵢ)
```

**Hard constraints:**
1. Each vessel assigned to exactly one berth
2. No two vessels overlap on the same berth at the same time
3. Vessel length ≤ berth quay length
4. Vessel draft ≤ berth depth
5. Vessel cargo type ∈ berth allowed cargo types
6. `sᵢ ≥ ETAᵢ` (cannot berth before arrival)

---

### 2. Data Layer

#### 2.1 CSV — `casa_history_data.csv` (12,829 rows, 39 columns)

This is your **training corpus and simulation source**. Here is how each column group maps to system components:

| Column group | Role in system |
|---|---|
| `Date/Heure Entrée au Port` | Vessel arrival event timestamp → ETA input |
| `Date/Heure Entree Mouillage` + `Sortie Mouillage` | Anchorage period → compute actual waiting time (target KPI) |
| `Date/Heure Accostage` + `Date App Quai` | Actual berthing window → ground truth for schedule |
| `Poste Accostage` | Actual berth assigned → ground truth label for berth selection |
| `Sejour Mouillage (Jours)` | **Primary optimization target**: waiting time to minimize |
| `Sejour Quai (Jours)` | **ML prediction target**: service duration at berth |
| `Sejour Port (Jours)` | Total port stay → derived KPI |
| `Longueur`, `Tirant Eau` | Physical vessel constraints → used in hard constraint filtering |
| `Jauge Brute` | Gross tonnage → proxy for vessel size in ML features |
| `Type Navire` | Vessel type (VRAQUIER, PORTE-CONTENEUR, etc.) → categorical ML feature |
| `Marchandise DAP Import/Export` | Cargo type → berth eligibility filter + ML feature |
| `Tonnage DAP Import/Export` | Cargo volume → strongest predictor of service time |
| `Nbr Remorque Accostage/Appareillage` | Tugboat usage → operational complexity proxy |
| `Provenances`, `Destinations` | Origin/destination → seasonality and route pattern features |

**Key data quality notes:**
- `Date Entree Mouillage` is null for ~7,568 rows (59%) — these are vessels that berthed directly without anchoring. Your model must handle both cases.
- `Sejour Mouillage`, `Sejour Quai`, `Sejour Port` are stored as objects — they will need parsing (likely `HH:MM:SS` or decimal string format).
- `Poste Accostage` is the ground truth berth — must be mapped to the JSON berth IDs for consistency.

#### 2.2 JSON — Port Infrastructure

The JSON gives you the **static digital twin** of the port. It defines:
- 9 terminals, each with operator, area, quay length, depth, equipment
- ~40+ named berths with cargo type restrictions
- Draft range: 7m–14m

This becomes your **Constraint Engine**: before any assignment, every vessel-berth pair is filtered through this structure.

---

### 3. System Architecture — Five Modules

#### Module 1: Data Pipeline

**Input:** Raw CSV + JSON  
**Output:** Clean feature matrix ready for ML + simulation

Steps:
1. Parse all date/time columns into proper `datetime` objects
2. Compute derived columns:
   - `waiting_time_hours` = `Date Accostage` − `Date Entrée au Port`
   - `service_time_hours` = `Date App Quai` − `Date Accostage`
   - `anchorage_flag` = 1 if `Date Entree Mouillage` is not null
3. Normalize `Poste Accostage` strings to match JSON berth IDs
4. One-hot encode `Type Navire` and `Marchandise DAP Import`
5. Build the **Berth Constraint Matrix**: a lookup table `berth_id → {max_length, max_depth, allowed_cargos, terminal}` from JSON

#### Module 2: Service Time Predictor (ML)

**Goal:** Given a new vessel's features, predict how long it will occupy a berth (`service_time_hours`).

**Model:** Gradient Boosted Trees (XGBoost/LightGBM) — more robust than linear regression for this data due to the categorical mix and outliers from weather delays. Linear regression is a valid baseline to compare against.

**Features X:**
```
Tonnage DAP Import, Tonnage DAP Export, Jauge Brute, Longueur,
Tirant Eau, Type Navire (encoded), Marchandise Import (encoded),
Nbr Remorque Accostage, month (seasonality), day_of_week
```

**Target y:** `service_time_hours`

**Training:** 80/20 split on historical data. Cross-validate with time-based split (train on years 1–3, test on year 4) to avoid data leakage.

**Output:** `predicted_service_time` for each incoming vessel → fed into the optimizer as the expected berth occupation duration.

#### Module 3: Constraint Engine

**Goal:** For each vessel in the queue, compute its set of **eligible berths**.

```
eligible_berths(vessel) = {
    berth ∈ B
    | berth.depth_m ≥ vessel.Tirant_Eau
    AND berth.quay_length_m ≥ vessel.Longueur
    AND vessel.cargo_type ∈ berth.cargo_types
}
```

This shrinks the search space dramatically before optimization. For example, a phosphate tanker is immediately restricted to berths 64 and 66 (OCP terminal). A container ship routes to berths 50–54 (SOMAPORT) or 70–83 (Marsa Maroc).

#### Module 4: Berth Allocation Optimizer (Genetic Algorithm)

**Goal:** Find the schedule `{(vessel, berth, start_time)}` that minimizes total waiting time across all vessels in the queue.

**GA Design:**

- **Chromosome:** A permutation of vessel indices + their assigned berths. Example for 5 vessels:
  ```
  [(V1→berth_72, t=10:00), (V2→berth_30, t=08:30), (V3→berth_52, t=14:00), ...]
  ```
- **Initialization:** Random valid assignments respecting hard constraints
- **Fitness:** `1 / (1 + Σ waiting_times)` — inversely proportional to total wait
- **Crossover:** Order-based crossover (OX) — preserves valid permutation structure
- **Mutation:** Swap two vessels' berth assignments or shift a start time by ±1 hour
- **Repair operator:** After each mutation/crossover, check hard constraints and fix violations by reassigning to the nearest eligible berth
- **Termination:** 500 generations or no improvement for 50 consecutive generations
- **Population size:** 100–200 chromosomes

**Alternative/complement:** A **greedy dispatch rule** as a warm-start: sort vessels by ETA, assign each to the earliest-available eligible berth. This gives the GA a good starting population and also serves as a fast baseline.

#### Module 5: Port Tracker (State Manager)

This is the system's real-time memory — a **berth timeline** structure:

```python
port_state = {
    "berth_72": [
        {"vessel": "V1", "start": datetime(2024,3,1,10,0), "end": datetime(2024,3,1,18,0)},
        {"vessel": "V3", "start": datetime(2024,3,1,20,0), "end": datetime(2024,3,2,6,0)},
    ],
    "berth_30": [...],
    ...
}
```

**Functions:**
- `get_next_available(berth_id, after_time)` → returns earliest free slot
- `assign(vessel, berth, start, end)` → adds a reservation
- `release(vessel)` → marks berth as free when vessel departs
- `get_port_snapshot()` → returns current occupancy for dashboard

---

### 4. Simulation Scenarios

Before live deployment, the system must be validated against these scenarios using historical data as ground truth:

| Scenario | Description | What it tests |
|---|---|---|
| **S1 — Baseline replay** | Feed historical ETAs one by one, compare system's assignments to actual `Poste Accostage` | Accuracy of berth selection |
| **S2 — High congestion** | Inject 2× normal vessel arrival rate (e.g., cereal import season) | Optimizer behavior under saturation |
| **S3 — Weather closure** | Mark port as closed for 12–48h, then release all waiting vessels at once | Burst handling, queue flushing |
| **S4 — Berth maintenance** | Remove 1–2 berths from availability (e.g., berth 72 under repair) | Constraint flexibility |
| **S5 — Late ETA** | Vessel announces ETA, then arrives 6h late | Schedule robustness, rescheduling trigger |
| **S6 — Priority vessel** | A vessel carries time-sensitive cargo (grains, bulk) and gets priority flag | Priority queue integration |
| **S7 — Multi-terminal conflict** | Two large container ships arrive simultaneously, only one berth available per terminal | Inter-terminal coordination |

---

### 5. KPIs to Track

| KPI | Formula | Source |
|---|---|---|
| Average waiting time | `mean(berthing_time − arrival_time)` | Port Tracker |
| Berth utilization rate | `sum(service_times) / (num_berths × time_horizon)` | Port Tracker |
| Schedule accuracy | `% of vessels berthed within 1h of predicted ETB` | Predictor output |
| ML prediction error | `RMSE of predicted vs actual service_time` | ML module |
| Queue length over time | `count(vessels waiting) per hour` | Simulation |

---

### 6. Technology Stack (recommended)

| Layer | Technology |
|---|---|
| Data pipeline | `pandas`, `numpy` |
| ML model | `scikit-learn` (LinearRegression baseline), `xgboost` (production) |
| Optimizer | `DEAP` library (Python GA framework) or custom |
| Port Tracker | Python class with `sortedcontainers.SortedList` for timeline |
| Simulation | `SimPy` (discrete event simulation) |
| Dashboard | `Streamlit` or `Dash` |
| Serialization | `joblib` (model), `json` (port state snapshots) |

---

Now here is the full Excalidraw diagram of the entire system:The `.excalidraw` file is ready. Import it at **excalidraw.com** → ☰ → Open.

---

Here is a summary of everything the diagram encodes, layer by layer, so you know exactly what you're approving before saying "implement":

**Layer 0 — Raw Data (3 sources):** The CSV (12,829 rows, historical vessel calls), the JSON (digital twin of 9 terminals and 40+ berths), and a live AIS/VTS feed for real-time ETAs. These are the only inputs the system ever needs.

**Layer 1 — Data Pipeline:** Five sequential steps that convert raw data into clean features. The most critical derived columns are `waiting_time_hours` (what you minimize) and `service_time_hours` (what ML predicts). The berth constraint matrix from JSON is also built here.

**Layer 2 — Intelligence Modules:** Two parallel modules. The ML predictor (XGBoost, baseline Linear Regression) takes 10 vessel features and outputs `predicted_service_time`. The Constraint Engine takes that same vessel and returns the list of physically and operationally eligible berths — this step alone eliminates 80–90% of invalid assignments before the optimizer even runs.

**Layer 3 — Genetic Algorithm Optimizer:** The core decision engine. Takes predicted service times + eligible berth lists, evolves a population of schedules over 500 generations, outputs the final `{vessel → berth, start_time}` assignment for every vessel in the queue. Objective: minimize total anchorage waiting time.

**Layer 4 — Port Tracker:** The system's memory. Holds a timeline for every berth, handles `assign/release/replan` events, and feeds the current port state back into the optimizer whenever a trigger fires (late arrival, weather, berth failure).

**Layer 5 — Simulation + Dashboard:** Seven scenarios validate the system before live use. The dashboard surfaces a Gantt chart, live berth occupancy, and four alert types for harbor master and terminal operators.

**Layer 6 — Stack:** `pandas`, `xgboost`, `DEAP`, `SimPy`, `Streamlit` — all Python, all open source.

When you're ready, just say **implement** and specify whether you want to start from the data pipeline, the ML module, or the GA optimizer.