# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:21:20 2026

Name: Sean San;ietro
CPS 485 Assignment 2
"""

from pathlib import Path
import numpy as np


DATA_GEN_OVERRIDE = r"C:\Users\Ssanp\Downloads\data_gen.py"
N = 2400
TRAIN_END = 1200
WINDOW = 365
SEEDS = (17, 29, 41, 53, 67)
HIDDEN_BETA = 650.0


def find_data_gen():
    path = Path(DATA_GEN_OVERRIDE)

    if not path.is_file():
        raise FileNotFoundError(f"data_gen.py not found: {path}")

    return path.resolve()


def generate(scenario, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    phase = 2 * np.pi * t / 365.25

    z = np.zeros(N)
    for i in range(N):
        z[i] = (0.55 * z[i - 1] if i else 0) + rng.normal(0, np.sqrt(1 - 0.55**2))
    z = np.clip(z, -2.25, 2.25)

    weekend = (t % 7 >= 5).astype(float)
    rain = rng.binomial(1, 0.16 + 0.05 * (1 - np.sin(phase))).astype(float)
    promotion = rng.binomial(1, 0.12, N).astype(float)
    temp_c = 14 + 12 * np.sin(phase) + 5.5 * z
    base = (2400 + 120 * np.sin(phase) + 60 * np.cos(phase)
            + 145 * weekend - 105 * rain + 210 * promotion)

    if scenario == "smooth_drift":
        beta_temp = 300 + 85 * np.sin(2 * np.pi * t / 1500)
        beta_weekend = 190 + 45 * np.cos(2 * np.pi * t / 1300)
        hidden_sign = np.zeros(N)
        mean = base + beta_temp * z + (beta_weekend - 145) * weekend + 55 * z * weekend
    else:
        hidden_sign = rng.choice([-1.0, 1.0], N)
        mean = base + HIDDEN_BETA * hidden_sign * z + 55 * z * weekend

    features = np.column_stack((
        np.ones(N), z, weekend, rain, promotion,
        np.sin(phase), np.cos(phase), z * weekend,
    ))
    rentals = rng.poisson(mean).astype(float)
    bayes_mean = mean if scenario == "smooth_drift" else base + 55 * z * weekend
    return features, rentals, mean, bayes_mean, temp_c


def fit_ridge(x, y):
    penalty = np.eye(x.shape[1]) * 2
    penalty[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def forecast(x, y, mean, bayes_mean):
    fixed = fit_ridge(x[:TRAIN_END], y[:TRAIN_END])
    size = N - TRAIN_END
    predictions = {
        "weekly_naive": np.empty(size),
        "fixed_ridge": np.empty(size),
        "rolling_ridge": np.empty(size),
        "bayes_oracle": bayes_mean[TRAIN_END:].copy(),
        "hindsight_oracle": mean[TRAIN_END:].copy(),
    }

    for j, day in enumerate(range(TRAIN_END, N)):
        predictions["weekly_naive"][j] = y[day - 7]
        predictions["fixed_ridge"][j] = x[day] @ fixed
        start = max(0, day - WINDOW)
        local_fit = fit_ridge(x[start:day], y[start:day])
        predictions["rolling_ridge"][j] = x[day] @ local_fit

    return {name: np.maximum(values, 0) for name, values in predictions.items()}


def scores(actual, prediction):
    error = prediction - actual
    rmse = np.sqrt(np.mean(error ** 2))
    r2 = 1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2)
    return rmse, r2


def main():
    generator_path = find_data_gen()
    if generator_path:
        print(f"Optional data_gen.py found: {generator_path}")
    else:
        print("data_gen.py not found; using the generators in this file.")

    results = {}
    hard_bounds = []
    model_names = ("weekly_naive", "fixed_ridge", "rolling_ridge", "bayes_oracle", "hindsight_oracle")
    for scenario in ("smooth_drift", "hidden_daily_switch"):
        results[scenario] = {name: [] for name in model_names}
        for seed in SEEDS:
            x, y, mean, bayes_mean, _ = generate(scenario, seed)
            if scenario == "hidden_daily_switch":
                hard_bounds.append(np.sqrt(np.mean(
                    bayes_mean[TRAIN_END:] + HIDDEN_BETA**2 * x[TRAIN_END:, 1]**2
                )))
            predictions = forecast(x, y, mean, bayes_mean)
            actual = y[TRAIN_END:]
            for name, prediction in predictions.items():
                results[scenario][name].append(scores(actual, prediction))

    labels = {
        "weekly_naive": "Weekly naive",
        "fixed_ridge": "Fixed ridge",
        "rolling_ridge": "Rolling ridge",
        "bayes_oracle": "Bayes oracle",
        "hindsight_oracle": "Hindsight oracle",
    }
    for scenario, models in results.items():
        print(f"\n{scenario}")
        print(f"{'Model':22s} {'RMSE (mean +/- SD)':22s} {'R2':>7s}")
        for name in model_names:
            runs = models[name]
            rmse = np.array([score[0] for score in runs])
            r2 = np.array([score[1] for score in runs])
            print(f"{labels[name]:22s} {rmse.mean():8.2f} +/- {rmse.std(ddof=1):6.2f} {r2.mean():7.3f}")

    print(f"\nHidden-switch theoretical Bayes RMSE: {np.mean(hard_bounds):.2f}")


if __name__ == "__main__":
    main()
