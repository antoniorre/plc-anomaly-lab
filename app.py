"""PLC Anomaly Lab — reproducible TLIGHT-inspired anomaly-detection demo.

This application deliberately uses a simulated process trace.  It implements
the research workflow (normal-only training, normalization, and three
unsupervised detectors) without claiming to reproduce the authors' raw HMI
logs or their numeric results.
"""

from __future__ import annotations

import csv
import io
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

app = Flask(__name__)

FEATURES = [
    "vehiculo_rojo", "vehiculo_amarillo", "vehiculo_verde",
    "peaton_rojo", "peaton_verde", "boton_peatonal",
    "solicitud_pendiente", "interbloqueo_seguro", "sync_amarillo",
    "retardo_peatonal", "fase_normalizada", "timer_verde_peatonal",
    "integridad_senales", "integridad_temporal",
]

SCENARIOS = {
    1: "Verde vehicular y verde peatonal simultáneos",
    2: "Todos los semáforos apagados",
    3: "Todos los semáforos encendidos",
    4: "Solicitud peatonal denegada",
    5: "Rojo vehicular y peatonal simultáneos",
    6: "Bit de sincronización amarillo manipulado",
    7: "Retardo/fase verde peatonal manipulados",
}

TEST_SPECS = {
    1: (5000, {1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02}),
    2: (7000, {3: 0.10}),
    3: (13130, {1: 0.10, 3: 0.10}),
    4: (15000, {6: 0.20, 7: 0.10}),
    5: (18270, {1: 0.05, 2: 0.10, 3: 0.10, 5: 0.05, 6: 0.05, 7: 0.15}),
}


def normal_trace(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate valid TLIGHT process-memory states at independently sampled times."""
    phase = rng.integers(0, 100, size=n)
    vehicle_green = (phase < 60).astype(float)
    vehicle_yellow = ((phase >= 60) & (phase < 70)).astype(float)
    vehicle_red = (phase >= 70).astype(float)
    pedestrian_green = (phase >= 70).astype(float)
    pedestrian_red = (phase < 70).astype(float)
    button = (rng.random(n) < 0.18).astype(float)
    # A valid request waits during vehicle green/yellow and is served during pedestrian green.
    pending = ((button == 1) & (phase < 70)).astype(float)
    # Diagnostic bits normally written by the safety/rule layer.  They are
    # independent monitored PLC addresses, not a target label.
    safe_interlock = np.ones(n, dtype=float)
    yellow_sync = vehicle_yellow.copy()
    pedestrian_delay = ((button == 1) & (phase < 70)).astype(float)
    phase_normalized = phase / 99.0
    pedestrian_timer = np.where(pedestrian_green == 1, (phase - 70) / 29.0, 0.0)

    return np.column_stack((
        vehicle_red, vehicle_yellow, vehicle_green, pedestrian_red,
        pedestrian_green, button, pending, safe_interlock, yellow_sync,
        pedestrian_delay, phase_normalized, pedestrian_timer,
        np.clip(rng.normal(0.96, 0.018, n), 0.88, 1.0),
        np.clip(rng.normal(0.95, 0.022, n), 0.84, 1.0),
    ))


def inject_scenario(x: np.ndarray, scenario: int, rng: np.random.Generator) -> np.ndarray:
    """Alter valid process states into one of the seven paper-inspired attacks."""
    z = x.copy()
    vr, vy, vg, pr, pg, button, pending, interlock, sync, delay, phase, timer, signal_ok, timing_ok = range(14)
    if scenario == 1:
        z[:, vg], z[:, pg] = 1, 1
        z[:, vr], z[:, pr], z[:, interlock], z[:, signal_ok] = 0, 0, 0, 0
    elif scenario == 2:
        z[:, [vr, vy, vg, pr, pg, interlock, signal_ok]] = 0
    elif scenario == 3:
        z[:, [vr, vy, vg, pr, pg]] = 1
        z[:, interlock], z[:, signal_ok] = 0, 0
    elif scenario == 4:
        # A persistent pedestrian request in a state that never transitions to green.
        z[:, button], z[:, pending], z[:, delay] = 1, 1, 1
        z[:, pg], z[:, pr], z[:, timer], z[:, signal_ok] = 0, 1, 0, 0
        z[:, phase] = rng.uniform(0.10, 0.55, len(z))
    elif scenario == 5:
        z[:, vr], z[:, pr] = 1, 1
        z[:, vg], z[:, pg], z[:, interlock], z[:, signal_ok] = 0, 0, 0, 0
    elif scenario == 6:
        # Timing bit disagrees with the observed vehicle-yellow lamp.
        z[:, sync], z[:, timing_ok] = 1 - z[:, vy], 0
    elif scenario == 7:
        # Pedestrian timing state disagrees with the phase / output state.
        z[:, delay] = 1 - z[:, delay]
        z[:, timer] = rng.uniform(0.65, 1.0, len(z))
        z[:, pending], z[:, timing_ok] = 1 - z[:, pending], 0
    else:
        raise ValueError("Escenario no reconocido")
    return z


def make_test_set(set_number: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if set_number not in TEST_SPECS:
        raise ValueError("Conjunto de prueba inválido")
    n, mixes = TEST_SPECS[set_number]
    rng = np.random.default_rng(seed + set_number * 1009)
    x = normal_trace(n, rng)
    labels = np.zeros(n, dtype=int)
    scenario_labels = np.zeros(n, dtype=int)
    chosen: list[int] = []
    for scenario, proportion in mixes.items():
        chosen.extend([scenario] * int(round(n * proportion)))
    chosen = chosen[:n]
    rng.shuffle(chosen)
    indices = rng.choice(n, size=len(chosen), replace=False)
    for index, scenario in zip(indices, chosen):
        x[index:index + 1] = inject_scenario(x[index:index + 1], scenario, rng)
        labels[index] = 1
        scenario_labels[index] = scenario
    order = rng.permutation(n)
    return x[order], labels[order], scenario_labels[order]


@dataclass
class OCNN:
    """Small NumPy one-class neural network following the OCNN hinge objective."""

    hidden_units: int = 32
    nu: float = 0.05
    # Kept deliberately compact so the interactive demonstration stays
    # responsive on a Render Free instance (0.1 CPU), too.
    epochs: int = 90
    learning_rate: float = 0.012
    random_state: int = 42

    def fit(self, x: np.ndarray) -> "OCNN":
        rng = np.random.default_rng(self.random_state)
        samples, dimensions = x.shape
        self.v_ = rng.normal(0, 0.22, size=(dimensions, self.hidden_units))
        self.w_ = rng.normal(0, 0.10, size=self.hidden_units)
        # Full-batch alternating minimisation: update r by its nu quantile,
        # then optimise weights with the one-class hinge loss.
        for _ in range(self.epochs):
            z = x @ self.v_
            h = np.maximum(z, 0)
            output = h @ self.w_
            self.r_ = float(np.quantile(output, self.nu))
            active = output < self.r_
            grad_output = -active.astype(float) / (self.nu * samples)
            grad_w = self.w_ + h.T @ grad_output
            grad_hidden = grad_output[:, None] * self.w_[None, :] * (z > 0)
            grad_v = self.v_ + x.T @ grad_hidden
            self.w_ -= self.learning_rate * grad_w
            self.v_ -= self.learning_rate * grad_v
        output = np.maximum(x @ self.v_, 0) @ self.w_
        self.r_ = float(np.quantile(output, self.nu))
        return self

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        output = np.maximum(x @ self.v_, 0) @ self.w_
        return self.r_ - output


def metric_bundle(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predicted = (scores > 0).astype(int)
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "alert_rate": round(float(predicted.mean()), 4),
    }


def normalise_for_histogram(scores: np.ndarray) -> list[float]:
    maximum = np.max(np.abs(scores))
    if maximum == 0:
        return [0.0] * len(scores)
    return (scores / maximum).round(4).tolist()


class Experiment:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = np.random.default_rng(seed)
        # A representative normal-only reference period.  This is sufficient
        # for the synthetic state space and avoids making a browser request
        # wait several minutes on a low-CPU hosting tier.
        self.training = normal_trace(1500, rng)
        self.scaler = MinMaxScaler().fit(self.training)
        scaled = self.scaler.transform(self.training)
        self.models: dict[str, Any] = {
            "OCSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(scaled),
            "OCNN": OCNN(random_state=seed).fit(scaled),
            "Isolation Forest": IsolationForest(
                n_estimators=120, max_samples=256, contamination=0.05,
                random_state=seed, n_jobs=-1,
            ).fit(scaled),
        }

    def score(self, name: str, raw: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(raw)
        model = self.models[name]
        if name == "OCNN":
            return model.anomaly_score(x)
        # scikit-learn returns positive values for normal points; reverse it
        # so that every detector emits a positive anomaly score.
        return -model.decision_function(x)

    def evaluate(self, set_number: int) -> dict[str, Any]:
        x, y, scenarios = make_test_set(set_number, self.seed)
        results: dict[str, Any] = {}
        for name in self.models:
            scores = self.score(name, x)
            bundle = metric_bundle(y, scores)
            # compact stratified samples of signed scores feed the histogram UI
            sample_count = min(500, len(x))
            chosen = np.linspace(0, len(x) - 1, sample_count, dtype=int)
            bundle["score_sample"] = normalise_for_histogram(scores[chosen])
            bundle["label_sample"] = y[chosen].tolist()
            results[name] = bundle
        return {
            "set": set_number,
            "samples": int(len(x)),
            "anomalies": int(y.sum()),
            "scenario_counts": {str(i): int((scenarios == i).sum()) for i in range(1, 8)},
            "models": results,
        }


_lock = threading.Lock()
_experiments: dict[int, Experiment] = {}


def get_experiment(seed: int) -> Experiment:
    if not 0 <= seed <= 999999:
        raise ValueError("La semilla debe estar entre 0 y 999999")
    with _lock:
        if seed not in _experiments:
            _experiments[seed] = Experiment(seed)
        return _experiments[seed]


@app.get("/")
def index() -> str:
    return render_template("index.html", features=FEATURES, scenarios=SCENARIOS)


@app.get("/api/metadata")
def metadata():
    return jsonify({
        "features": FEATURES,
        "scenarios": SCENARIOS,
        "test_specs": {str(k): {"samples": v[0], "mix": v[1]} for k, v in TEST_SPECS.items()},
        "disclaimer": "Simulación TLIGHT reproducible; no contiene los registros HMI originales del artículo.",
    })


@app.post("/api/run")
def run_experiment():
    body = request.get_json(silent=True) or {}
    try:
        seed = int(body.get("seed", 2026))
        selected_sets = body.get("sets", [1, 2, 3, 4, 5])
        selected_sets = [int(item) for item in selected_sets]
        experiment = get_experiment(seed)
        return jsonify({
            "seed": seed,
            "training_samples": len(experiment.training),
            "results": [experiment.evaluate(number) for number in selected_sets],
            "notice": "Los valores son resultados de esta simulación y variarán con la semilla; no son cifras copiadas del artículo.",
        })
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/inspect")
def inspect_state():
    body = request.get_json(silent=True) or {}
    try:
        seed = int(body.get("seed", 2026))
        scenario = body.get("scenario", "normal")
        rng = np.random.default_rng(seed + 7919)
        raw = normal_trace(1, rng)
        label = 0
        if scenario != "normal":
            scenario = int(scenario)
            raw = inject_scenario(raw, scenario, rng)
            label = 1
        experiment = get_experiment(seed)
        scores = {name: float(experiment.score(name, raw)[0]) for name in experiment.models}
        return jsonify({
            "state": {feature: round(float(value), 3) for feature, value in zip(FEATURES, raw[0])},
            "expected_anomaly": bool(label),
            "scenario": scenario,
            "scores": {name: {"score": round(score, 5), "decision": "ANOMALÍA" if score > 0 else "Normal"} for name, score in scores.items()},
        })
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/download/test-set-<int:set_number>.csv")
def download_dataset(set_number: int):
    try:
        seed = int(request.args.get("seed", 2026))
        x, labels, scenarios = make_test_set(set_number, seed)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([*FEATURES, "is_anomaly", "scenario"])
    for values, label, scenario in zip(x, labels, scenarios):
        writer.writerow([*(f"{value:.5f}" for value in values), int(label), int(scenario)])
    memory = io.BytesIO(output.getvalue().encode("utf-8"))
    return send_file(memory, mimetype="text/csv", as_attachment=True, download_name=f"tlight_test_set_{set_number}.csv")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "plc-anomaly-lab"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
