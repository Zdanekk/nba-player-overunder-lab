from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import norm

# Importy działające i dla "python -m src.*", i dla "uvicorn --app-dir src"
try:
    from features.minutes_model import predict_minutes
    from features.form_ewma import make_rates
except Exception:
    from src.features.minutes_model import predict_minutes
    from src.features.form_ewma import make_rates

EPS = 1e-6

def _row_pred(
    player_id: int,
    opponent_id: int | None,
    market: str,
    line: float,
    gamelogs: pd.DataFrame,
    mins_df: pd.DataFrame,
    factors: dict[str, dict[int, float]] | None,
    alpha: float = 0.30,
):
    # 1) oczekiwane minuty
    mrow = mins_df.loc[mins_df["player_id"] == player_id]
    E_min = float(mrow["E_min"].values[0]) if not mrow.empty else 28.0
    sd_min = float(mrow["sd_min"].values[0]) if not mrow.empty else 5.0

    # 2) stawki per minute (EWMA+shrink)
    r = make_rates(gamelogs, player_id)
    market = (market or "").upper()
    if market == "PTS":
        r_hat, sd_r = float(r["r_pts"]), float(r["sd_r_pts"])
    elif market == "REB":
        r_hat, sd_r = float(r["r_reb"]), float(r["sd_r_reb"])
    else:
        r_hat, sd_r = float(r["r_ast"]), float(r["sd_r_ast"])

    # 3) baza: Normal przez "delta method"
    r_hat = max(r_hat, 0.0)
    mu = max(0.0, E_min * r_hat)
    var = (r_hat**2) * (sd_min**2) + (E_min**2) * (sd_r**2)
    sd = float(np.sqrt(max(var, EPS)))

    # 4) matchup (opponent-adjust) – łagodzony przez alpha
    if factors and opponent_id is not None:
        if market == "PTS":
            f = float(factors.get("PTS", {}).get(int(opponent_id), 1.0))
        elif market == "REB":
            f = float(factors.get("REB", {}).get(int(opponent_id), 1.0))
        else:
            f = float(factors.get("AST", {}).get(int(opponent_id), 1.0))
        adj = 1.0 + alpha * (f - 1.0)       # np. f=1.10 -> adj=1.03 gdy alpha=0.30
        adj = float(np.clip(adj, 0.6, 1.4)) # zabezpieczenie przed ekstremami
        mu *= adj
        sd *= adj

    # 5) percentyle, P(Over), fair line (≈ mediana w Normalu)
    p10 = max(0.0, float(norm.ppf(0.10, loc=mu, scale=sd)))
    p50 = max(0.0, float(norm.ppf(0.50, loc=mu, scale=sd)))
    p90 = max(0.0, float(norm.ppf(0.90, loc=mu, scale=sd)))
    z = (line - mu) / sd if sd > EPS else 1e9
    p_over = float(1.0 - norm.cdf(z))
    fair_line = round(mu * 2) / 2.0

    return mu, sd, p10, p50, p90, p_over, fair_line

def predict(
    props_df: pd.DataFrame,
    gamelogs: pd.DataFrame,
    allowed_factors: pd.DataFrame | None = None,
    alpha: float = 0.30,
) -> pd.DataFrame:
    """
    Prognozy (mu, sigma, p10/50/90, p_over, fair_line_model) dla PTS/REB/AST.
    Jeżeli podasz allowed_factors (team_id, f_pts/f_reb/f_ast) — uwzględnimy matchup.
    """
    mins_df = predict_minutes(gamelogs)

    # Słownik faktorów {market -> {team_id: factor}}
    factors = None
    if allowed_factors is not None and not allowed_factors.empty:
        factors = {
            "PTS": dict(zip(allowed_factors["team_id"].astype(int), allowed_factors["f_pts"])),
            "REB": dict(zip(allowed_factors["team_id"].astype(int), allowed_factors["f_reb"])),
            "AST": dict(zip(allowed_factors["team_id"].astype(int), allowed_factors["f_ast"])),
        }

    rows = []
    for _, r in props_df.iterrows():
        mu, sd, p10, p50, p90, p_over, fair_line = _row_pred(
            int(r["player_id"]),
            int(r["opponent_id"]) if pd.notna(r["opponent_id"]) else None,
            str(r["market"]).upper(),
            float(r["line"]),
            gamelogs,
            mins_df,
            factors,
            alpha=alpha,
        )
        rows.append({
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
            "model": "Baseline",
        })
    return pd.DataFrame(rows)
