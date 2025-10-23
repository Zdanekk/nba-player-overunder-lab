from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import norm

# importy przyjazne dla --app-dir src i python -m src.*
try:
    from features.minutes_model import predict_minutes
    from features.form_ewma import make_rates
except Exception:
    from src.features.minutes_model import predict_minutes
    from src.features.form_ewma import make_rates

EPS = 1e-6

def _fit_lin(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Dopasuj y = a + b*x (OLS) i zwróć (a, b, RMSE)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.c_[np.ones_like(x), x]
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - (coef[0] + coef[1] * x)
        rmse = float(np.sqrt(np.mean(resid**2))) if len(y) > 1 else max(1.0, float(np.std(y) or 1.0))
        return float(coef[0]), float(coef[1]), rmse
    except Exception:
        return 0.0, np.nan, float(np.std(y) or 1.0)

def _row_pred_glm(
    player_id: int,
    opponent_id: int | None,
    market: str,
    line: float,
    gamelogs: pd.DataFrame,
    mins_df: pd.DataFrame,
    factors: dict[str, dict[int, float]] | None,
    alpha: float = 0.30,
):
    g = gamelogs[gamelogs["player_id"] == player_id].sort_values("game_date").tail(15)
    market = (market or "").upper()

    if g.empty:
        # fallback: EWMA rate * E[min]
        r = make_rates(gamelogs, player_id)
        E_min = float(mins_df.loc[mins_df["player_id"] == player_id, "E_min"].fillna(28.0).values[0])
        if market == "PTS":
            mu = E_min * float(r["r_pts"])
        elif market == "REB":
            mu = E_min * float(r["r_reb"])
        else:
            mu = E_min * float(r["r_ast"])
        sd = max(1.0, mu * 0.35)
    else:
        if market == "PTS":
            y = g["pts"].values
        elif market == "REB":
            y = g["reb"].values
        else:
            y = g["ast"].values
        x = g["min"].values
        a, b, rmse = _fit_lin(y, x)
        mrow = mins_df.loc[mins_df["player_id"] == player_id]
        E_min = float(mrow["E_min"].values[0]) if not mrow.empty else 28.0
        mu = float(a + b * E_min)
        sd = max(0.5, float(rmse))

    # opponent adjust
    if factors and opponent_id is not None:
        if market == "PTS":
            f = float(factors.get("PTS", {}).get(int(opponent_id), 1.0))
        elif market == "REB":
            f = float(factors.get("REB", {}).get(int(opponent_id), 1.0))
        else:
            f = float(factors.get("AST", {}).get(int(opponent_id), 1.0))
        adj = 1.0 + alpha * (f - 1.0)
        adj = float(np.clip(adj, 0.6, 1.4))
        mu *= adj
        sd *= adj

    # P(Over), percentyle, fair line
    p10 = max(0.0, float(norm.ppf(0.10, loc=mu, scale=sd)))
    p50 = max(0.0, float(norm.ppf(0.50, loc=mu, scale=sd)))
    p90 = max(0.0, float(norm.ppf(0.90, loc=mu, scale=sd)))
    z = (line - mu) / sd if sd > EPS else 1e9
    p_over = float(1.0 - norm.cdf(z))
    fair_line = round(mu * 2) / 2.0

    return mu, sd, p10, p50, p90, p_over, fair_line

def predict_glm(
    props_df: pd.DataFrame,
    gamelogs: pd.DataFrame,
    allowed_factors: pd.DataFrame | None = None,
    alpha: float = 0.30,
) -> pd.DataFrame:
    mins_df = predict_minutes(gamelogs)

    factors = None
    if allowed_factors is not None and not allowed_factors.empty:
        factors = {
            "PTS": dict(zip(allowed_factors["team_id"].astype(int), allowed_factors["f_pts"])),
            "REB": dict(zip(allowed_factors["team_id"].astype(int), allowed_factors["f_reb"])),
            "AST": dict(zip(allowed_factors["team_id"].astype(int), allowed_factors["f_ast"])),
        }

    out = []
    for _, r in props_df.iterrows():
        mu, sd, p10, p50, p90, p_over, fair_line = _row_pred_glm(
            int(r["player_id"]),
            int(r["opponent_id"]) if pd.notna(r["opponent_id"]) else None,
            str(r["market"]).upper(),
            float(r["line"]),
            gamelogs, mins_df, factors, alpha
        )
        out.append({
            "game_id": r["game_id"],
            "player_id": r["player_id"],
            "player_name": r["player_name"],
            "team_id": r["team_id"],
            "opponent_id": r["opponent_id"],
            "market": str(r["market"]).upper(),
            "line": float(r["line"]),
            "mu": round(mu, 2),
            "sigma": round(sd, 2),
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p_over": round(p_over, 3),
            "fair_line_model": round(fair_line, 1),
            "model": "GLM",
        })
    return pd.DataFrame(out)
