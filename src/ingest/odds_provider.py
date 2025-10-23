from __future__ import annotations
import pandas as pd
from datetime import date
from pathlib import Path

def fetch_props(d: date) -> pd.DataFrame:
    """
    Czyta propsy z data/props_YYYY-MM-DD.csv.
    Jeśli brak pliku na dany dzień, bierze najnowszy dostępny props_*.csv i
    dopisuje kolumnę '_source_file' (dla przejrzystości).
    """
    base = Path(__file__).resolve().parents[2]
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    p = data_dir / f"props_{d.isoformat()}.csv"
    if not p.exists():
        candidates = sorted(data_dir.glob("props_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"Brak {p} i brak jakichkolwiek plików props_*.csv w {data_dir}."
            )
        p = candidates[-1]  # ostatni dostępny
    df = pd.read_csv(p)
    required = {"game_id","player_id","player_name","team_id","opponent_id","market","line","book","ts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn w CSV: {missing} (plik: {p})")

    df["market"] = df["market"].astype(str).str.upper().str.strip()
    df = df[df["market"].isin(["PTS","REB","AST"])].copy()
    df["_source_file"] = str(p)
    return df
