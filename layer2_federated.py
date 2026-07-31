"""
Layer 2 — Federated linear regression

Implements weighted FedAvg, FedProx, and optional personalization using
a common feature coordinate system and deterministic mini-batch SGD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import (
    AGE_GROUPS,
    FEATURES,
    FL_CFG,
    FLConfig,
)
from layer1_simulation import HospitalData


@dataclass(frozen=True)
class SharedScaler:
    """Common feature scaling parameters used by every client."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def compute_federated_scaler(
    hospitals: dict[str, HospitalData],
) -> SharedScaler:
    """
    Compute global mean and variance from client-level sufficient statistics.

    Only counts, feature sums, and squared-feature sums are aggregated.
    Patient-level rows are not exchanged.
    """

    if not hospitals:
        raise ValueError("At least one hospital is required.")

    n_features = next(
        iter(hospitals.values())
    ).X_train.shape[1]

    total_count = 0
    total_sum = np.zeros(
        n_features,
        dtype=float,
    )
    total_sum_sq = np.zeros(
        n_features,
        dtype=float,
    )

    for hospital in hospitals.values():
        X = np.asarray(
            hospital.X_train,
            dtype=float,
        )

        if X.ndim != 2 or X.shape[1] != n_features:
            raise ValueError(
                f"Incompatible feature shape for {hospital.name}: {X.shape}"
            )

        total_count += len(X)
        total_sum += X.sum(axis=0)
        total_sum_sq += np.square(X).sum(axis=0)

    if total_count == 0:
        raise ValueError("No training observations are available.")

    mean = total_sum / total_count
    variance = (
        total_sum_sq / total_count
        - np.square(mean)
    )
    variance = np.maximum(
        variance,
        0.0,
    )

    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0

    return SharedScaler(
        mean=mean,
        scale=scale,
    )


class LinearModel:
    """Linear regression optimized by deterministic mini-batch SGD."""

    def __init__(
        self,
        n_features: int,
        random_state: int,
    ) -> None:
        self.n_features = n_features
        self.coef_ = np.zeros(
            n_features,
            dtype=float,
        )
        self.intercept_ = 0.0
        self.rng = np.random.default_rng(
            random_state
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        global_weights: Optional[dict] = None,
        proximal_mu: float = 0.0,
    ) -> None:
        """
        Train locally from the current global model.

        For FedProx, both coefficients and intercept are penalized relative
        to the global parameter vector.
        """

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of observations."
            )
        if len(y) == 0:
            raise ValueError(
                "Local training data cannot be empty."
            )
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )
        if lr <= 0:
            raise ValueError(
                "learning rate must be positive."
            )
        if proximal_mu < 0:
            raise ValueError(
                "proximal_mu cannot be negative."
            )

        if global_weights is not None:
            self.set_weights(global_weights)

        global_coef = None
        global_intercept = None

        if (
            proximal_mu > 0
            and global_weights is not None
        ):
            global_coef = np.asarray(
                global_weights["coef"],
                dtype=float,
            ).copy()
            global_intercept = float(
                global_weights["intercept"]
            )

        n_samples = len(y)
        batch_size = min(
            32,
            n_samples,
        )

        for _ in range(epochs):
            permutation = self.rng.permutation(
                n_samples
            )

            for start in range(
                0,
                n_samples,
                batch_size,
            ):
                indices = permutation[
                    start : start + batch_size
                ]
                X_batch = X[indices]
                y_batch = y[indices]

                residual = (
                    self.predict(X_batch)
                    - y_batch
                )

                grad_coef = (
                    X_batch.T @ residual
                ) / len(indices)
                grad_intercept = float(
                    residual.mean()
                )

                if (
                    proximal_mu > 0
                    and global_coef is not None
                    and global_intercept is not None
                ):
                    grad_coef += proximal_mu * (
                        self.coef_ - global_coef
                    )
                    grad_intercept += proximal_mu * (
                        self.intercept_
                        - global_intercept
                    )

                self.coef_ -= lr * grad_coef
                self.intercept_ -= (
                    lr * grad_intercept
                )

    def get_weights(self) -> dict:
        return {
            "coef": self.coef_.copy(),
            "intercept": float(self.intercept_),
        }

    def set_weights(
        self,
        weights: dict,
    ) -> None:
        coef = np.asarray(
            weights["coef"],
            dtype=float,
        )

        if coef.shape != (self.n_features,):
            raise ValueError(
                f"Expected coefficient shape {(self.n_features,)}, "
                f"received {coef.shape}."
            )

        self.coef_ = coef.copy()
        self.intercept_ = float(
            weights["intercept"]
        )


class FLClient:
    """One hospital client in the federated network."""

    def __init__(
        self,
        hospital: HospitalData,
        scaler: SharedScaler,
        cfg: FLConfig,
        random_state: int,
    ) -> None:
        self.name = hospital.name
        self.hospital = hospital
        self.cfg = cfg
        self.scaler = scaler

        # All clients now use the same coordinate system.
        self.X_train = scaler.transform(
            hospital.X_train
        )
        self.X_test = scaler.transform(
            hospital.X_test
        )
        self.y_train = np.asarray(
            hospital.y_train,
            dtype=float,
        )
        self.y_test = np.asarray(
            hospital.y_test,
            dtype=float,
        )

        self.model = LinearModel(
            n_features=self.X_train.shape[1],
            random_state=random_state,
        )

    def train(
        self,
        global_weights: Optional[dict],
        proximal_mu: float,
    ) -> dict:
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.cfg.local_epochs,
            lr=self.cfg.learning_rate,
            global_weights=global_weights,
            proximal_mu=proximal_mu,
        )
        return self.model.get_weights()

    def fine_tune(
        self,
        global_weights: dict,
    ) -> None:
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.cfg.personalized_finetune_epochs,
            lr=self.cfg.learning_rate * 0.4,
            global_weights=global_weights,
            proximal_mu=0.0,
        )

    def evaluate(self) -> dict:
        predictions = self.model.predict(
            self.X_test
        )
        return _metrics(
            self.y_test,
            predictions,
            prefix=self.name,
        )

    def evaluate_subgroups(
        self,
        minimum_size: int = 5,
    ) -> dict:
        predictions = self.model.predict(
            self.X_test
        )
        metadata = self.hospital.meta_test
        output: dict[str, float] = {}

        for age_group in AGE_GROUPS:
            mask = (
                metadata["AgeGroup"] == age_group
            ).to_numpy()

            if int(mask.sum()) < minimum_size:
                continue

            output[f"age_{age_group}"] = float(
                mean_absolute_error(
                    self.y_test[mask],
                    predictions[mask],
                )
            )

        for gender in ("M", "F"):
            mask = (
                metadata["Gender"] == gender
            ).to_numpy()

            if int(mask.sum()) < minimum_size:
                continue

            output[f"gender_{gender}"] = float(
                mean_absolute_error(
                    self.y_test[mask],
                    predictions[mask],
                )
            )

        return output


class FedAvgServer:
    """Weighted parameter aggregation server."""

    def __init__(
        self,
        n_features: int,
        random_state: int,
    ) -> None:
        self.global_model = LinearModel(
            n_features=n_features,
            random_state=random_state,
        )

    def aggregate(
        self,
        client_weights: dict[str, dict],
        client_sizes: dict[str, int],
        importance_weights: dict[str, float],
    ) -> dict:
        if not client_weights:
            raise ValueError(
                "No client updates were supplied."
            )

        effective_weights: dict[str, float] = {}

        for name in client_weights:
            importance = float(
                importance_weights.get(
                    name,
                    1.0,
                )
            )
            if importance <= 0:
                raise ValueError(
                    f"Importance weight for {name} must be positive."
                )

            effective_weights[name] = (
                float(client_sizes[name])
                * importance
            )

        total_weight = sum(
            effective_weights.values()
        )
        if total_weight <= 0:
            raise ValueError(
                "Total aggregation weight must be positive."
            )

        first_weights = next(
            iter(client_weights.values())
        )

        aggregated_coef = np.zeros_like(
            np.asarray(
                first_weights["coef"],
                dtype=float,
            )
        )
        aggregated_intercept = 0.0

        for name, weights in client_weights.items():
            normalized_weight = (
                effective_weights[name]
                / total_weight
            )

            aggregated_coef += (
                normalized_weight
                * np.asarray(
                    weights["coef"],
                    dtype=float,
                )
            )
            aggregated_intercept += (
                normalized_weight
                * float(weights["intercept"])
            )

        global_weights = {
            "coef": aggregated_coef,
            "intercept": aggregated_intercept,
        }

        self.global_model.set_weights(
            global_weights
        )
        return global_weights


def run_federated_rounds(
    clients: dict[str, FLClient],
    server: FedAvgServer,
    cfg: FLConfig,
    importance_weights: Optional[dict] = None,
) -> dict:
    """Run FedAvg or FedProx for the configured number of rounds."""

    if importance_weights is None:
        importance_weights = {
            name: 1.0
            for name in clients
        }

    client_sizes = {
        name: client.hospital.n_train
        for name, client in clients.items()
    }

    proximal_mu = (
        cfg.fed_prox_mu
        if cfg.strategy == "fedprox"
        else 0.0
    )

    global_weights = (
        server.global_model.get_weights()
    )
    round_mae: list[float] = []

    for round_index in range(
        1,
        cfg.n_rounds + 1,
    ):
        local_weights = {
            name: client.train(
                global_weights=global_weights,
                proximal_mu=proximal_mu,
            )
            for name, client in clients.items()
        }

        global_weights = server.aggregate(
            client_weights=local_weights,
            client_sizes=client_sizes,
            importance_weights=importance_weights,
        )

        for client in clients.values():
            client.model.set_weights(
                global_weights
            )

        current_mae = float(
            np.mean(
                [
                    mean_absolute_error(
                        client.y_test,
                        client.model.predict(
                            client.X_test
                        ),
                    )
                    for client in clients.values()
                ]
            )
        )
        round_mae.append(current_mae)

        print(
            f"Round {round_index:>3}/{cfg.n_rounds}: "
            f"average MAE={current_mae:.3f}"
        )

    if cfg.strategy == "personalized":
        for client in clients.values():
            client.fine_tune(
                global_weights
            )

    final_metrics = {
        name: client.evaluate()
        for name, client in clients.items()
    }
    subgroup_metrics = {
        name: client.evaluate_subgroups(
            minimum_size=5
        )
        for name, client in clients.items()
    }

    return {
        "round_mae": round_mae,
        "final_metrics": final_metrics,
        "subgroup_metrics": subgroup_metrics,
        "global_weights": global_weights,
        "strategy": cfg.strategy,
        "fed_prox_mu": proximal_mu,
        "importance_weights": dict(
            importance_weights
        ),
    }


def run_layer2(
    hospitals: dict[str, HospitalData],
    cfg: FLConfig = FL_CFG,
    importance_weights: Optional[dict] = None,
) -> tuple[
    dict[str, FLClient],
    FedAvgServer,
    dict,
]:
    """Train and evaluate the federated prediction environment."""

    if not hospitals:
        raise ValueError(
            "Layer 2 requires at least one hospital."
        )

    shared_scaler = compute_federated_scaler(
        hospitals
    )

    clients: dict[str, FLClient] = {}

    # Stable deterministic client seeds.
    for client_index, (name, hospital) in enumerate(
        sorted(hospitals.items())
    ):
        clients[name] = FLClient(
            hospital=hospital,
            scaler=shared_scaler,
            cfg=cfg,
            random_state=(
                cfg.random_state
                + 1000 * client_index
            ),
        )

    server = FedAvgServer(
        n_features=len(FEATURES),
        random_state=cfg.random_state,
    )

    results = run_federated_rounds(
        clients=clients,
        server=server,
        cfg=cfg,
        importance_weights=importance_weights,
    )

    results["shared_scaler"] = {
        "mean": shared_scaler.mean.tolist(),
        "scale": shared_scaler.scale.tolist(),
    }

    return clients, server, results


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> dict:
    return {
        f"{prefix}_mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        f"{prefix}_rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        f"{prefix}_r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }