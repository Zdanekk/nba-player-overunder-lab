from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd

try:
    from ingest.odds_provider import fetch_props
    from ingest.nba_api_fetch import fetch_player_gamelogs
    from features.opponent_adjust import build_allowed_factors
    from features.minutes_model import predict_minutes
    from models.m_baseline_ewma import predict as predict_baseline
    from models.m_glm_opponent import predict_glm
    from models.m_negbin_count import predict_negbin
    from ensemble.stacker import sample_mixture, disagreement
except Exception:
    from src.ingest.odds_provider import fetch_props
    from src.ingest.nba_api_fetch import fetch_player_gamelogs
    from src.features.opponent_adjust import build_allowed_factors
    from src.features.minutes_model import predict_minutes
    from src.models.m_baseline_ewma import predict as predict_baseline
    from src.models.m_glm_opponent import predict_glm
    from src.models.m_negbin_count import predict_negbin
    from src.ensemble.stacker import sample_mixture, disagreement

def _recommend(row) -> tuple[str, float, str, float]:
    mu, sd, line, p, disc = row["mu_consensus"], row["sigma_consensus"], row["line"], row["p_over_consensus"], row["disagreement"]
    sd = max(sd, 1e-6)
    z = (mu - line) / sd
    if p >= 0.60 and z >= 0.60:    rec = "OVER"
    elif p <= 0.40 and z <= -0.60: rec = "UNDER"
    else:                           rec = "PASS"
    conf_score = float(max(p, 1-p) * (1.0 / (1.0 + disc)))
    bucket = "high" if conf_score >= 0.70 else ("medium" if conf_score >= 0.60 else "low")
    return rec, float(z), bucket, round(conf_score, 3)

def run_compare_day(d: date) -> str:
    base = Path(__file__).resolve().parents[1]
    out_dir = base / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    props = fetch_props(d)
    players_hint = props[["player_id","player_name","team_id"]].drop_duplicates()
    opponents = sorted(props["opponent_id"].dropna().astype(int).unique().tolist())

    gamelogs = fetch_player_gamelogs(last_n_days=180, players_hint=players_hint)
    allowed = build_allowed_factors(gamelogs, window_games=10, required_team_ids=opponents)
    mins_df = predict_minutes(gamelogs)[["player_id","E_min","sd_min","minutes_quality"]]

    a = predict_baseline(props, gamelogs, allowed_factors=allowed, alpha=0.30)   # model="Baseline"
    b = predict_glm(props, gamelogs, allowed_factors=allowed, alpha=0.30)        # model="GLM"
    c = predict_negbin(props, gamelogs, allowed_factors=allowed, alpha=0.30)     # model="NegBin"

    both = pd.concat([a, b, c], ignore_index=True)

    rows = []
    for (gid, pid, mkt), grp in both.groupby(["game_id","player_id","market"]):
        # consensus via sampling
        cons = sample_mixture(grp, n_total=3000, random_state=42)
        disc = disagreement(grp)

        row = {
            "game_id": gid,
            "player_id": pid,
            "player_name": grp["player_name"].iloc[0],
            "team_id": grp["team_id"].iloc[0],
            "opponent_id": grp["opponent_id"].iloc[0],
            "market": mkt,
            "line": float(grp["line"].iloc[0]),

            "E_min": float(mins_df.loc[mins_df["player_id"]==pid, "E_min"].fillna(28.0).values[0]),
            "sd_min": float(mins_df.loc[mins_df["player_id"]==pid, "sd_min"].fillna(5.0).values[0]),

            "mu_baseline": float(grp.loc[grp["model"]=="Baseline","mu"].iloc[0]),
            "p_over_baseline": float(grp.loc[grp["model"]=="Baseline","p_over"].iloc[0]),
            "fair_line_baseline": float(grp.loc[grp["model"]=="Baseline","fair_line_model"].iloc[0]),

            "mu_glm": float(grp.loc[grp["model"]=="GLM","mu"].iloc[0]),
            "p_over_glm": float(grp.loc[grp["model"]=="GLM","p_over"].iloc[0]),
            "fair_line_glm": float(grp.loc[grp["model"]=="GLM","fair_line_model"].iloc[0]),

            "mu_negbin": float(grp.loc[grp["model"]=="NegBin","mu"].iloc[0]),
            "p_over_negbin": float(grp.loc[grp["model"]=="NegBin","p_over"].iloc[0]),
            "fair_line_negbin": float(grp.loc[grp["model"]=="NegBin","fair_line_model"].iloc[0]),

            "mu_consensus": cons["mu_consensus"],
            "sigma_consensus": cons["sigma_consensus"],
            "p_over_consensus": cons["p_over_consensus"],
            "fair_line_consensus": cons["fair_line_consensus"],
            "edge_points": cons["edge_points"],   # dodatnia → OVER lean
            "disagreement": round(disc, 3)
        }

        rec, zc, bucket, conf = _recommend(row)
        row.update({"z_consensus": round(zc,3), "recommendation": rec,
                    "confidence_bucket": bucket, "confidence_score": conf})
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = out_dir / f"props_compare_{d.isoformat()}.csv"
    out.to_csv(out_path, index=False)
    return str(out_path)

if __name__ == "__main__":
    print(run_compare_day(date.today()))
