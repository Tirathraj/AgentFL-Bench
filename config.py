"""
config.py
─────────
Central configuration for the PICU Federated Learning Fairness Pipeline.
Edit values here — all three layers read from these dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────

DATA_PATH  = "/mnt/fedlearn/agentic/master_picu_data.csv" #Path("master_picu_data.csv")   # real Sainte-Justine data
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
#  Dataset schema
# ─────────────────────────────────────────────

FEATURES = ["BodyTemp", "Age_Months"]
TARGET   = "HeartRate"
META_COLS = ["AgeGroup", "Gender"]

AGE_GROUPS = [
    "0-1 month",      "1-6 months",     "6-12 months",
    "1-3 years old",  "3-5 years old",  "5-7 years old",
    "7-9 years old",  "9-11 years old", "11-14 years old",
    "14-18 years old",
]


# ─────────────────────────────────────────────
#  Layer 1 — Bias injection parameters
# ─────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Controls how SickKids simulated data is generated."""

    # Scenario 1 — age distribution shift
    # Fraction of newborns (0-1 month) to duplicate → simulates NICU-heavy site
    age_shift_ratio: float = 0.4

    # Scenario 2 — temperature measurement bias
    # Systematic offset in °C → simulates miscalibrated thermometer
    temp_bias: float = -0.5

    # Scenario 3 — device noise
    # Gaussian std added to HeartRate (bpm) and BodyTemp (°C * 0.05)
    device_noise_std: float = 2.0

    # Scenario 4 — physiological relationship shift
    # Flat HR offset applied to 3–9 yr age groups (bpm) → different patient mix
    physio_hr_shift: float = 5.0

    # Scenario 5 — data imbalance
    # SickKids retains this fraction of SJ data size
    sk_data_fraction: float = 0.6

    seed: int = 42


# ─────────────────────────────────────────────
#  Layer 2 — Federated Learning parameters
# ─────────────────────────────────────────────

@dataclass
class FLConfig:
    """Controls federated learning training."""

    n_rounds:      int   = 10      # number of FL communication rounds
    local_epochs:  int   = 60      # SGD epochs per client per round
    learning_rate: float = 0.01    # SGD learning rate

    # FedProx proximal penalty  (only used when strategy == "fedprox")
    fed_prox_mu: float = 0.01

    # Importance weights per client (1.0 = equal weight in aggregation)
    client_weights: dict = field(default_factory=lambda: {
        "sainte_justine": 1.0,
        "sickkids":        1.0,
    })

    # FL strategy: "fedavg" | "fedprox" | "personalized"
    strategy: str = "fedavg"

    # Extra local fine-tuning epochs for personalized FL
    personalized_finetune_epochs: int = 20

    test_size: float = 0.2
    random_state: int = 42


# ─────────────────────────────────────────────
#  Layer 3 — Agentic system parameters
# ─────────────────────────────────────────────
@dataclass
class AgentConfig:
    provider: str = "vllm"

    vllm_base_url: str = "http://127.0.0.1:8601/v1"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_api_key: str = "EMPTY"

    temperature: float = 0.0
    max_iterations: int = 12
    harm_mae_threshold: float = 20.0
    hospital_gap_threshold: float = 3.0
# ─────────────────────────────────────────────
#  Instantiated defaults (imported by other modules)
# ─────────────────────────────────────────────

SIM_CFG   = SimulationConfig()
FL_CFG    = FLConfig()
AGENT_CFG = AgentConfig()
