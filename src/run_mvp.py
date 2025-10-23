from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd

try:
    from ingest.odds_provider import fetch_props
    from ingest.nba_api_fetch import fetch_player_gamelogs
    from features.opponent_adjust import build_allowed_factors
    from models.m_baseline_ewma import predict
except Exception:
    from src.ingest.odds_provider import fetch_props
    from src.ingest.nba_api_fetch import fetch_player_gamelogs
    from src.features.opponent_adjust import build_allowed_factors
    from src.models.m_baseline_ewma import predict

def run_day(d: date) -> str:
    base = Path(__file__).resolve().parents[1]
    out_dir = base / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    props = fetch_props(d)

    # HINT: listy graczy i drużyn do fallbacku
    players_hint = props[["player_id","player_name","team_id"]].drop_duplicates()
    opponents = sorted(props["opponent_id"].dropna().astype(int).unique().tolist())

    gamelogs = fetch_player_gamelogs(last_n_days=180, players_hint=players_hint)

    allowed = build_allowed_factors(gamelogs, window_games=10, required_team_ids=opponents)

    preds = predict(props, gamelogs, allowed_factors=allowed, alpha=0.30)
    out_path = out_dir / f"props_predictions_{d.isoformat()}.csv"
    preds.to_csv(out_path, index=False)
    return str(out_path)

if __name__ == "__main__":
    print(run_day(date.today()))
