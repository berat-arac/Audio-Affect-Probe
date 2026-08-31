from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_results"
SEEDS = [42, 1337, 2026, 31415]


def main() -> None:
    df = pd.read_csv(REF / "final_four_seed_comparison.csv")

    print("Four seed model means")
    means = (
        df.groupby("model")[["valence_ccc", "arousal_ccc", "mean_ccc"]]
        .mean()
        .loc[["dual", "salience", "dual_split"]]
    )
    print(means.to_string(float_format=lambda x: f"{x:.6f}"))

    dual = df[df.model == "dual"].set_index("seed").loc[SEEDS]
    salience = df[df.model == "salience"].set_index("seed").loc[SEEDS]
    split = df[df.model == "dual_split"].set_index("seed").loc[SEEDS]

    salience_delta = salience[["valence_ccc", "arousal_ccc", "mean_ccc"]] - dual[["valence_ccc", "arousal_ccc", "mean_ccc"]]
    split_delta = split[["valence_ccc", "arousal_ccc", "mean_ccc"]] - dual[["valence_ccc", "arousal_ccc", "mean_ccc"]]

    print()
    print("Mean salience minus dual delta")
    print(salience_delta.mean().to_string(float_format=lambda x: f"{x:.6f}"))

    print()
    print("Mean dual_split minus dual delta")
    print(split_delta.mean().to_string(float_format=lambda x: f"{x:.6f}"))

    expected = {
        "dual": (0.268514, 0.714884, 0.491699),
        "salience": (0.224656, 0.733948, 0.479302),
        "dual_split": (0.260509, 0.720680, 0.490595),
    }
    for model, vals in expected.items():
        got = means.loc[model].to_numpy()
        if max(abs(got[i] - vals[i]) for i in range(3)) > 1e-6:
            raise AssertionError(f"Reference mean mismatch for {model}: {got}")

    print()
    print("REFERENCE_RESULTS_OK")


if __name__ == "__main__":
    main()
