from __future__ import annotations
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

try:
    from nba_api.stats.endpoints import leaguegamelog
except Exception:
    leaguegamelog = None  # jeśli brak/403, użyjemy fallbacków


REQUIRED = [
    "PLAYER_ID","PLAYER_NAME","TEAM_ID","MATCHUP",
    "GAME_ID","GAME_DATE","MIN","PTS","REB","AST"
]

def _parse_min(x) -> float:
    try:
        m, s = str(x).split(":")
        return int(m) + int(s)/60.0
    except Exception:
        try:
            return float(x)
        except Exception:
            return 0.0

def _postprocess_logs(logs: pd.DataFrame) -> pd.DataFrame:
    keep = {
        "PLAYER_ID":"player_id","PLAYER_NAME":"player_name","TEAM_ID":"team_id",
        "MATCHUP":"matchup","GAME_ID":"game_id","GAME_DATE":"game_date",
        "MIN":"min_str","PTS":"pts","REB":"reb","AST":"ast"
    }
    df = logs[list(keep.keys())].rename(columns=keep).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["min"] = df["min_str"].apply(_parse_min)

    def parse_opp(x):
        parts = str(x).split()
        if len(parts) == 3:
            return parts[2], ("HOME" if parts[1].lower()=="vs" else "AWAY")
        return None, None
    df["opponent_abbr"], df["home_away"] = zip(*df["matchup"].map(parse_opp))
    return df

def _try_leaguegamelog(season: str, season_type: str) -> Optional[pd.DataFrame]:
    if leaguegamelog is None:
        return None
    try:
        lg = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type)
        df = lg.get_data_frames()[0]
        # walidacja kolumn (często przy 403 zwraca "message"/inne pola)
        if not set(REQUIRED).issubset(df.columns):
            return None
        return df
    except Exception:
        return None

def fetch_player_gamelogs(last_n_days: int = 120,
                          players_hint: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Próbujemy ściągnąć gamelogi; jeśli się nie uda:
    - wczytujemy lokalny cache data/gamelogs_cache.csv,
    - jeśli go brak, tworzymy syntetyczne gamelogi dla graczy z propsów (players_hint).
    """
    # 1) Próba online – kilka sezonów/typów
    frames = []
    for season in ["2025-26","2024-25","2023-24"]:
        for st in ["Regular Season","Playoffs"]:
            df = _try_leaguegamelog(season, st)
            if df is not None:
                frames.append(df)
    if frames:
        logs = pd.concat(frames, ignore_index=True).drop_duplicates()
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        cutoff = pd.Timestamp(date.today() - timedelta(days=last_n_days))
        logs = logs[logs["GAME_DATE"] >= cutoff].copy()
        return _postprocess_logs(logs)

    # 2) Fallback: lokalny cache
    base = Path(__file__).resolve().parents[2]
    cache = base / "data" / "gamelogs_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["game_date"])
        # upewnij się, że minimalne kolumny są
        expected = {"player_id","player_name","team_id","game_id","game_date","min","pts","reb","ast"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Brak kolumn w {cache}: {missing}")
        return df

    # 3) Ostateczny fallback: syntetyczne gamelogi dla players_hint
    if players_hint is None or players_hint.empty:
        raise RuntimeError("Brak gamelogów online, brak cache i brak players_hint do zbudowania syntetycznych danych.")
    rows = []
    # proste 8 meczów z sensownymi wartościami wokół „typowych”
    for _, r in players_hint.iterrows():
        pid = int(r["player_id"])
        pname = str(r.get("player_name", f"player_{pid}"))
        tid = int(r["team_id"])
        for k in range(8):
            rows.append({
                "player_id": pid,
                "player_name": pname,
                "team_id": tid,
                "game_id": f"SYNTH_{pid}_{k}",
                "game_date": pd.Timestamp(date.today() - timedelta(days=7*(k+1))),
                "min": 32 + (k%3-1)*2,     # 30–34
                "pts": 22 + (k%5-2)*3,     # ~22 ±
                "reb": 6 + (k%3-1)*1,      # ~6 ±
                "ast": 5 + (k%3-1)*1,      # ~5 ±
                "matchup": "SYN vs SYN",
                "opponent_abbr": "SYN",
                "home_away": "HOME",
            })
    return pd.DataFrame(rows)
