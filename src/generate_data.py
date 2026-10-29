import numpy as np
import pandas as pd
from pathlib import Path

def simulate_kpis(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Genere n pas de temps de KPIs 5G et injecte 8 anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    df = pd.DataFrame({
        "rsrp":    -85 + 3 * np.sin(t / 500) + rng.normal(0, 1.5, n),
        "sinr":     18 + rng.normal(0, 1.2, n),
        "loss":     np.abs(rng.normal(0.2, 0.1, n)),
        "traffic":  50 + 10 * np.sin(t / 300) + rng.normal(0, 3, n),
        "cpu":      40 + rng.normal(0, 4, n),
    })
    label = np.zeros(n, dtype=int)
    for f in rng.choice(np.arange(2000, n - 100), 8, replace=False):
        w = 30                        # fenetre de degradation
        ramp = np.linspace(0, 1, w)
        df.loc[f - w:f - 1, "sinr"]  -= 10 * ramp
        df.loc[f - w:f - 1, "loss"]  += 5 * ramp
        df.loc[f - w:f - 1, "cpu"]   += 30 * ramp
        label[f - w:f] = 1
    df["anomaly"] = label
    return df

if __name__ == "__main__":
    out = Path("data"); out.mkdir(exist_ok=True)
    df = simulate_kpis()
    df.to_csv(out / "kpis.csv", index=False)
    print(df.head())
    print(df["anomaly"].value_counts())