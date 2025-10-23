from __future__ import annotations
import pandas as pd
import numpy as np

def _rate_per_min(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] / df["min"].replace(0, np.nan)

def make_rates(gamelogs: pd.DataFrame, player_id: int) -> dict:
    """
    Zwraca EWMA i season_mean dla PTS/REB/AST per-minute dla jednego gracza.
    """
    g = gamelogs[gamelogs["player_id"]==player_id].sort_values("game_date").copy()
    if g.empty:
        return {"r_pts":0.0,"r_reb":0.0,"r_ast":0.0, "sd_r_pts":0.2,"sd_r_reb":0.2,"sd_r_ast":0.2}
    for c in ["pts","reb","ast"]:
        g[f"{c}_rpm"] = _rate_per_min(g, c)
    # sezonowa średnia (po filtrze)
    season_means = {
        "pts": g["pts_rpm"].mean(),
        "reb": g["reb_rpm"].mean(),
        "ast": g["ast_rpm"].mean()
    }
    # EWMA (alpha ~ okno 10 gier)
    ewma = {
        "pts": g["pts_rpm"].ewm(alpha=0.2, adjust=False).mean().iloc[-1],
        "reb": g["reb_rpm"].ewm(alpha=0.2, adjust=False).mean().iloc[-1],
        "ast": g["ast_rpm"].ewm(alpha=0.2, adjust=False).mean().iloc[-1],
    }
    # shrink do sezonu (empirical Bayes lite)
    k = 10  # siła shrinku
    n = len(g)
    w = min(1.0, n/(n+k))  # im mniej danych, tym większy shrink do sezonu
    r_pts = w*ewma["pts"] + (1-w)*season_means["pts"]
    r_reb = w*ewma["reb"] + (1-w)*season_means["reb"]
    r_ast = w*ewma["ast"] + (1-w)*season_means["ast"]
    # odchylenie z ostatnich 10 gier
    tail = g.tail(10)
    sd = {
        "pts": float(tail["pts_rpm"].std(ddof=1) or 0.2),
        "reb": float(tail["reb_rpm"].std(ddof=1) or 0.2),
        "ast": float(tail["ast_rpm"].std(ddof=1) or 0.2),
    }
    return {"r_pts":float(r_pts), "r_reb":float(r_reb), "r_ast":float(r_ast),
            "sd_r_pts":sd["pts"], "sd_r_reb":sd["reb"], "sd_r_ast":sd["ast"]}
