from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import norm, nbinom

def mix_normals(df_models: pd.DataFrame) -> pd.DataFrame:
    # (zostawiamy – użyteczne jako szybki fallback)
    mus = df_models["mu"].values.astype(float)
    sig = df_models["sigma"].values.astype(float)
    line = float(df_models["line"].iloc[0])
    w = np.ones_like(mus) / len(mus)
    mu_mix = float(np.sum(w * mus))
    var_mix = float(np.sum(w * (sig**2 + mus**2)) - mu_mix**2)
    sd_mix = float(np.sqrt(max(var_mix, 1e-6)))
    z = (line - mu_mix) / sd_mix if sd_mix > 0 else 1e9
    p_over = float(1.0 - norm.cdf(z))
    return pd.DataFrame([{
        "mu_consensus": round(mu_mix, 2),
        "sigma_consensus": round(sd_mix, 2),
        "p_over_consensus": round(p_over, 3),
        "fair_line_consensus": round(mu_mix*2)/2.0
    }])

def sample_mixture(df_models: pd.DataFrame, n_total: int = 3000, random_state: int = 42) -> dict:
    """
    Mieszamy rozkłady modeli (Normal dla Baseline/GLM, NegBin dla C).
    Zwracamy consensus metryki + fair line (mediana próbek) i edge.
    """
    rng = np.random.default_rng(random_state)
    line = float(df_models["line"].iloc[0])
    models = df_models["model"].tolist()
    n = max(200, int(n_total / max(1, len(models))))
    samples = []

    for _, r in df_models.iterrows():
        mu, sd = float(r["mu"]), float(r["sigma"])
        mname = str(r["model"])
        if mname in ("Baseline","GLM"):
            s = rng.normal(mu, sd, size=n)
            s = np.clip(s, 0.0, None)
        elif mname == "NegBin":
            # odtwórz przybliżone parametry NB z mu i sd -> phi
            var = sd**2
            phi = max((var - mu) / (mu**2), 1e-6) if mu > 1e-6 else 0.15
            rnb = 1.0 / phi
            pnb = rnb / (rnb + max(mu, 1e-6))
            s = nbinom.rvs(rnb, pnb, size=n, random_state=rng)
        else:
            s = rng.normal(mu, sd, size=n)
        samples.append(s)

    mix = np.concatenate(samples) if samples else np.array([])
    if mix.size == 0:
        return {"mu_consensus": float("nan"), "sigma_consensus": float("nan"),
                "p_over_consensus": float("nan"), "fair_line_consensus": float("nan"),
                "edge_points": float("nan")}

    mu_c = float(np.mean(mix))
    sd_c = float(np.std(mix, ddof=1))
    p_over = float(np.mean(mix > line))
    fair_line = float(np.median(mix))
    edge = fair_line - line  # dodatni => lean OVER

    return {
        "mu_consensus": round(mu_c, 2),
        "sigma_consensus": round(sd_c, 2),
        "p_over_consensus": round(p_over, 3),
        "fair_line_consensus": round(round(fair_line*2)/2.0, 1),
        "edge_points": round(edge, 2)
    }

def disagreement(df_models: pd.DataFrame) -> float:
    mu_min, mu_max = df_models["mu"].min(), df_models["mu"].max()
    avg_sigma = max(1e-6, float(df_models["sigma"].mean()))
    return float(abs(mu_max - mu_min) / avg_sigma)
