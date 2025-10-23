from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import nbinom

# importy przyjazne dla --app-dir src i python -m src.*
try:
    from features.minutes_model import predict_minutes
    from features.form_ewma import make_rates
except Exception:
    from src.features.minutes_model import predict_minutes
    from src.features.form_ewma import make_rates

def _estimate_phi_from_history(g: pd.DataFrame, market: str, max_tail: int = 20) -> float:
    """
    Szacunek overdispersion phi metodą momentów na ostatnich meczach gracza.
    Var = mu + phi*mu^2  =>  phi = max((Var - mu)/mu^2, 0)
    """
    tail = g.sort_values("game_date").tail(max_tail)
    if tail.empty:
        return 0.15
    x = tail["pts"].values if market == "PTS" else (tail["reb"].values if market == "REB" else tail["ast"].values)
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if len(x) > 1 else (mu + 1.0)
    if mu <= 1e-9:
        return 0.15
    phi = max((var - mu) / (mu**2), 0.0)
    # Ogranicz do rozsądnych widełek
    return float(np.clip(phi, 0.02, 0.60))

def _nb_params(mu: float, phi: float):
    """
    Parametry NegBin w konwencji SciPy:
      r = 1/phi,  p = r / (r + mu)
    """
    if phi <= 0:
        phi = 1e-6
    r = 1.0 / phi
    p = r / (r + max(mu, 1e-6))
    return r, p

def _row_pred_negbin(player_id: int, opponent_id: int, market: str, line: float,
                     gamelogs: pd.DataFrame, mins_df: pd.DataFrame,
                     factors: dict[str, dict[int, float]] | None,
                     alpha: float = 0.30):
    """
    Mean mu bazuje na: E[min] * rate_per_min (EWMA+shrink) * opponent-adjust.
    Rozkład: Negative Binomial z phi z historii gracza.
    """
    # E[min]
    mrow = mins_df.loc[mins_df["player_id"] == player_id]
    E_min = float(mrow["E_min"].values[0]) if not mrow.empty else 28.0

    # stawki per minute
    rdict = make_rates(gamelogs, player_id)
    if market == "PTS":
        rate = rdict["r_pts"]
        series_col = "pts"
    elif market == "REB":
        rate = rdict["r_reb"]
        series_col = "reb"
    else:
        rate = rdict["r_ast"]
        series_col = "ast"

    # bazowe mu
    mu = max(0.0, E_min * rate)

    # opponent-adjust
    if factors is not None and opponent_id is not None:
        f = 1.0
        if market == "PTS": f = factors.get("PTS", {}).get(int(opponent_id), 1.0)
        elif market == "REB": f = factors.get("REB", {}).get(int(opponent_id), 1.0)
        else: f = factors.get("AST", {}).get(int(opponent_id), 1.0)
        mu *= (1.0 + alpha*(f - 1.0))

    # nadrozproszenie z historii
    g = gamelogs[gamelogs["player_id"] == player_id]
    phi = _estimate_phi_from_history(g, market)

    # parametry NB
    r, p = _nb_params(mu, phi)

    # P(Over)
    # dla linii x.5: P(X > line) = 1 - CDF(floor(line))
    thr = int(np.floor(line))
    p_over = float(1.0 - nbinom.cdf(thr, r, p))

    # mediana (fair line modelu)
    med = float(nbinom.ppf(0.5, r, p))
    fair_line = round(med * 2) / 2.0  # zaokrąglij do .5 jak u buków

    # sigma (przybliżone sd)
    var = mu + phi * (mu**2)
    sd = float(np.sqrt(max(var, 1e-6)))

    # percentyle 10/50/90
    p10 = float(nbinom.ppf(0.10, r, p))
    p50 = med
    p90 = float(nbinom.ppf(0.90, r, p))

    return mu, sd, p10, p50, p90, p_over, fair_line

def predict_negbin(props_df: pd.DataFrame, gamelogs: pd.DataFrame,
                   allowed_factors: pd.DataFrame | None = None,
                   alpha: float = 0.30) -> pd.DataFrame:
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
        mu, sd, p10, p50, p90, p_over, fair_line = _row_pred_negbin(
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
            "market": r["market"],
            "line": r["line"],
            "mu": round(mu, 2),
            "sigma": round(sd, 2),
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p_over": round(p_over, 3),
            "fair_line_model": round(fair_line, 1),
            "model": "NegBin"
        })
    return pd.DataFrame(out)
