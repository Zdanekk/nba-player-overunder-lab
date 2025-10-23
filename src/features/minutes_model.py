from __future__ import annotations
import pandas as pd
import numpy as np

def predict_minutes(
    gamelogs: pd.DataFrame,
    window: int = 10,
    ewma_alpha: float = 0.35,
    hard_floor: float = 12.0,
    hard_cap: float = 42.0,
) -> pd.DataFrame:
    """
    Prognoza minut (E_min, sd_min) per player:
    - sort po dacie,
    - winsoryzacja minut (5-95 pct) dla stabilności,
    - miks: 0.6*EWMA + 0.4*średnia z ostatnich `window` gier,
    - sd z ostatnich `window` gier (zabezpieczenia + clip),
    - clip E_min do [hard_floor, hard_cap].
    """
    if gamelogs is None or gamelogs.empty:
        return pd.DataFrame(columns=["player_id","E_min","sd_min","player_name","team_id","minutes_quality"])

    gl = gamelogs.sort_values(["player_id", "game_date"]).copy()

    def _one_player(df: pd.DataFrame) -> pd.Series:
        s = df["min"].astype(float).copy()
        n = len(s)
        # winsoryzacja
        if n >= 4:
            q05, q95 = s.quantile(0.05), s.quantile(0.95)
            s = s.clip(lower=q05, upper=q95)
        # komponenty
        ewma = s.ewm(alpha=ewma_alpha, adjust=False).mean().iloc[-1]
        tail = s.tail(window)
        roll = tail.mean()
        sd = float(tail.std(ddof=1)) if len(tail) >= 2 else float(s.std(ddof=1) or 4.0)
        # miks i bezpieczne ograniczenia
        E = 0.6 * float(ewma) + 0.4 * float(roll)
        E = float(np.clip(E, hard_floor, hard_cap))
        sd = float(np.clip(sd, 2.0, 8.0))
        quality = "low" if n < 5 else "ok"
        return pd.Series({"E_min": E, "sd_min": sd, "minutes_quality": quality})

    agg = gl.groupby(["player_id","player_name","team_id"], as_index=False).apply(_one_player)
    # pandas <=2.2 zachowuje stare indeksy po apply; ustandaryzuj kolumny:
    if "player_id" not in agg.columns:
        agg = agg.reset_index()
    return agg[["player_id","E_min","sd_min","player_name","team_id","minutes_quality"]]
