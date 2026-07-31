"""
Layer 1 — Data simulation

Loads the CHU Sainte-Justine PICU dataset, partitions patients into two
disjoint client cohorts, and applies controlled perturbations to the second
cohort to create a counterfactually simulated hospital.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    FEATURES,
    META_COLS,
    SIM_CFG,
    TARGET,
    SimulationConfig,
)


@dataclass
class HospitalData:
    """Train/test data and subgroup metadata for one FL client."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    n_train: int = 0
    n_test: int = 0

    def __post_init__(self) -> None:
        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

    def summary(self) -> str:
        return (
            f"{self.name}: train={self.n_train}, "
            f"test={self.n_test}, features={self.X_train.shape[1]}"
        )


def load_real_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load, clean, and keep one record per patient."""

    df = pd.read_csv(path)

    required_columns = [
        "PatientID",
        *FEATURES,
        TARGET,
        *META_COLS,
    ]
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    df = df.drop_duplicates(
        subset=["PatientID"],
        keep="first",
    )
    df = df.dropna(
        subset=[*FEATURES, TARGET, *META_COLS]
    )

    return df.reset_index(drop=True)


def split_client_cohorts(
    df: pd.DataFrame,
    simulated_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition patients into disjoint real and simulated client cohorts.

    The simulated hospital is derived only from its assigned patient subset,
    preventing the same patient from appearing in both clients.
    """

    if not 0.0 < simulated_fraction < 1.0:
        raise ValueError(
            "simulated_fraction must be between 0 and 1."
        )

    sj_df, simulated_source_df = train_test_split(
        df,
        test_size=simulated_fraction,
        random_state=random_state,
        stratify=df["AgeGroup"],
    )

    return (
        sj_df.reset_index(drop=True),
        simulated_source_df.reset_index(drop=True),
    )


def make_hospital_data(
    df: pd.DataFrame,
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> HospitalData:
    """Create a stratified train/test split for one client."""

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["AgeGroup"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return HospitalData(
        name=name,
        X_train=train_df[FEATURES].to_numpy(dtype=float),
        X_test=test_df[FEATURES].to_numpy(dtype=float),
        y_train=train_df[TARGET].to_numpy(dtype=float),
        y_test=test_df[TARGET].to_numpy(dtype=float),
        meta_train=train_df[META_COLS].reset_index(drop=True),
        meta_test=test_df[META_COLS].reset_index(drop=True),
    )


def inject_age_shift(
    df: pd.DataFrame,
    ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Oversample newborns to create an age-distribution shift."""

    newborns = df[df["AgeGroup"] == "0-1 month"]
    n_extra = int(len(newborns) * ratio)

    if n_extra <= 0 or newborns.empty:
        return df.copy()

    extra = newborns.sample(
        n=n_extra,
        replace=True,
        random_state=int(rng.integers(1_000_000)),
    )

    return pd.concat(
        [df, extra],
        ignore_index=True,
    )


def inject_temp_bias(
    df: pd.DataFrame,
    bias: float,
) -> pd.DataFrame:
    """Apply a systematic body-temperature offset."""

    result = df.copy()
    result["BodyTemp"] += bias
    return result


def inject_device_noise(
    df: pd.DataFrame,
    noise_std: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add controlled Gaussian noise to temperature and heart rate."""

    result = df.copy()
    n_rows = len(result)

    result["HeartRate"] += rng.normal(
        0.0,
        noise_std,
        n_rows,
    )
    result["BodyTemp"] += rng.normal(
        0.0,
        noise_std * 0.05,
        n_rows,
    )

    result["HeartRate"] = result["HeartRate"].clip(
        30,
        250,
    )
    result["BodyTemp"] = result["BodyTemp"].clip(
        32,
        42,
    )

    return result


def inject_physio_shift(
    df: pd.DataFrame,
    hr_shift: float,
) -> pd.DataFrame:
    """Apply a controlled heart-rate shift to children aged 3–9 years."""

    result = df.copy()

    affected = result["AgeGroup"].isin(
        [
            "3-5 years old",
            "5-7 years old",
            "7-9 years old",
        ]
    )

    result.loc[affected, "HeartRate"] += hr_shift
    result["HeartRate"] = result["HeartRate"].clip(
        30,
        250,
    )

    return result


def inject_data_imbalance(
    df: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Subsample the simulated client to create data imbalance."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "sk_data_fraction must be in the interval (0, 1]."
        )

    n_samples = min(
        len(df),
        max(80, int(len(df) * fraction)),
    )

    return df.sample(
        n=n_samples,
        replace=False,
        random_state=int(rng.integers(1_000_000)),
    ).reset_index(drop=True)


def simulate_sickkids(
    source_df: pd.DataFrame,
    cfg: SimulationConfig = SIM_CFG,
) -> pd.DataFrame:
    """Apply the five controlled perturbations sequentially."""

    rng = np.random.default_rng(cfg.seed)
    result = source_df.copy()

    result = inject_age_shift(
        result,
        cfg.age_shift_ratio,
        rng,
    )
    result = inject_temp_bias(
        result,
        cfg.temp_bias,
    )
    result = inject_device_noise(
        result,
        cfg.device_noise_std,
        rng,
    )
    result = inject_physio_shift(
        result,
        cfg.physio_hr_shift,
    )
    result = inject_data_imbalance(
        result,
        cfg.sk_data_fraction,
        rng,
    )

    return result


def run_layer1(
    cfg: SimulationConfig = SIM_CFG,
    test_size: float = 0.2,
    simulated_patient_fraction: float = 0.40,
) -> dict[str, HospitalData]:
    """
    Load the cohort and create two disjoint FL clients.

    Sainte-Justine uses the unmodified real-data partition. The second
    partition is transformed to create the simulated client.
    """

    print("\n" + "─" * 56)
    print("LAYER 1 — Data Simulation")
    print("─" * 56)

    full_df = load_real_data()
    print(
        f"Loaded {len(full_df)} unique patients."
    )

    sj_df, simulated_source_df = split_client_cohorts(
        full_df,
        simulated_fraction=simulated_patient_fraction,
        random_state=cfg.seed,
    )

    sk_df = simulate_sickkids(
        simulated_source_df,
        cfg,
    )

    # Defensive check: the original patient cohorts must be disjoint.
    overlap = set(sj_df["PatientID"]).intersection(
        set(simulated_source_df["PatientID"])
    )
    if overlap:
        raise RuntimeError(
            f"Patient leakage detected across clients: {len(overlap)} IDs."
        )

    hospitals = {
        "sainte_justine": make_hospital_data(
            sj_df,
            name="sainte_justine",
            test_size=test_size,
            random_state=cfg.seed,
        ),
        "sickkids": make_hospital_data(
            sk_df,
            name="sickkids",
            test_size=test_size,
            random_state=cfg.seed,
        ),
    }

    for hospital in hospitals.values():
        print(hospital.summary())

    return hospitals