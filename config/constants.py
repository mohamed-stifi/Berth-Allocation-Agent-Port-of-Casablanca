"""
Central registry for every literal value used across the system.
No magic numbers or strings anywhere else — import from here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT          = Path(__file__).resolve().parents[1]
DATA_DIR              = PROJECT_ROOT / "data"
CSV_PATH              = DATA_DIR / "casa_history_data.csv"
JSON_PATH             = DATA_DIR / "port_details.json"
MODEL_DIR             = PROJECT_ROOT / "ml"
SERVICE_TIME_MODEL    = MODEL_DIR / "service_time_predictor.pkl"

# ── CSV column names ─────────────────────────────────────────────────────────
COL_VESSEL_NAME          = "Navire"
COL_VESSEL_TYPE          = "Type Navire"
COL_GROSS_TONNAGE        = "Jauge Brute"
COL_DRAFT_ENTRY_FORE     = "Tirant Eau Entree Avant"
COL_DRAFT_ENTRY_AFT      = "Tirant Eau Entree Arriere"
COL_LENGTH               = "Longueur"
COL_DRAFT                = "Tirant Eau"
COL_DATE_PORT_ENTRY      = "Date Entrée au Port"
COL_TIME_PORT_ENTRY      = "Heure Entrée au Port"
COL_DATE_ANCHORAGE_IN    = "Date Entree Mouillage"
COL_TIME_ANCHORAGE_IN    = "Heure Entree Mouillage"
COL_DATE_ANCHORAGE_OUT   = "Date Sortie Mouillage"
COL_TIME_ANCHORAGE_OUT   = "Heure Sortie Mouillage"
COL_DATE_BERTHING        = "Date Accostage"
COL_TIME_BERTHING        = "Heure Accostage"
COL_DATE_UNBERTHING      = "Date App Quai"
COL_TIME_UNBERTHING      = "Heure App Quai"
COL_BERTH_ID             = "Poste Accostage"
COL_DATE_DEPARTURE       = "Date Appareillage Port"
COL_TIME_DEPARTURE       = "Heure Appareillage Port"
COL_STAY_ANCHORAGE_DAYS  = "Sejour Mouillage (Jours)"
COL_STAY_QUAY_DAYS       = "Sejour Quai (Jours)"
COL_STAY_PORT_DAYS       = "Sejour Port (Jours)"
COL_CARGO_IMPORT         = "Marchandise DAP Import"
COL_CARGO_EXPORT         = "Marchandise DAP Export"
COL_TONNAGE_IMPORT       = "Tonnage DAP Import"
COL_TONNAGE_EXPORT       = "Tonnage DAP Export"
COL_TUGS_BERTHING        = "Nbr Remorque Accostage"
COL_TUGS_DEPARTURE       = "Nbr Remorque Appareillage"
COL_ORIGIN               = "Provenances"

# ── Derived column names (added by pipeline) ─────────────────────────────────
COL_ARRIVAL_DT           = "arrival_datetime"
COL_BERTHING_DT          = "berthing_datetime"
COL_UNBERTHING_DT        = "unberthing_datetime"
COL_WAITING_HOURS        = "waiting_time_hours"
COL_SERVICE_HOURS        = "service_time_hours"
COL_ANCHORAGE_FLAG       = "anchored_before_berthing"
COL_ARRIVAL_MONTH        = "arrival_month"
COL_ARRIVAL_DOW          = "arrival_day_of_week"

# ── ML model settings ────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT_RATIO   = 0.80          # 80 % train, 20 % test
MIN_SERVICE_HOURS        = 0.5           # discard sub-30-min stays (data noise)
MAX_SERVICE_HOURS        = 30 * 24       # discard stays > 30 days (outliers)
RANDOM_STATE             = 42

# ── Genetic Algorithm settings ───────────────────────────────────────────────
GA_POPULATION_SIZE       = 150
GA_MAX_GENERATIONS       = 500
GA_NO_IMPROVE_PATIENCE   = 50
GA_CROSSOVER_PROBABILITY = 0.85
GA_MUTATION_PROBABILITY  = 0.15
GA_TOURNAMENT_SIZE       = 5

# ── Port physical limits ─────────────────────────────────────────────────────
PORT_MIN_DRAFT_M         = 7.0
PORT_MAX_DRAFT_M         = 14.0

# ── Simulation settings ──────────────────────────────────────────────────────
SIM_CONGESTION_MULTIPLIER        = 2.0   # S2: double arrival rate
SIM_WEATHER_CLOSURE_HOURS        = 24    # S3: port closed 24 h
SIM_LATE_ETA_DELAY_HOURS         = 6     # S5: vessel arrives 6 h late
SIM_PRIORITY_FLAG                = "PRIORITY"
SIM_MAINTENANCE_BERTHS           = ["72", "30"]   # S4: taken offline

# ── Replan triggers ───────────────────────────────────────────────────────────
LATE_ARRIVAL_THRESHOLD_HOURS     = 2.0   # trigger replan if delay > 2 h

# ── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_HOST           = "0.0.0.0"
DASHBOARD_PORT           = 8501
CONGESTION_ALERT_VESSELS = 10            # alert if queue > 10 vessels
