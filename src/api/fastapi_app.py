from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from datetime import date
from pathlib import Path
import pandas as pd

# Próba importu runnerów zarówno dla --app-dir src, jak i bez
try:
    from run_mvp import run_day
    from run_compare import run_compare_day
except Exception:
    from src.run_mvp import run_day           # fallback, gdy import wyżej się nie uda
    from src.run_compare import run_compare_day

app = FastAPI(title="NBA Player Over/Under Lab")

@app.get("/")
def root():
    return {
        "service": "NBA Player Over/Under Lab",
        "endpoints": {
            "/health": "status",
            "/docs": "Swagger UI",
            "/predict/today": "Baseline (JSON)",
            "/predict/{date_str}": "Baseline dla daty (JSON)",
            "/props/compare/today": "Baseline vs GLM vs NegBin + Consensus (JSON)",
            "/props/compare/{date_str}": "j.w. dla daty (JSON)",
            "/props/compare/table/today": "Widok tabeli HTML",
            "/props/compare/table/{date_str}": "Widok tabeli HTML dla daty"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def _respond(path: str, cols: list[str]):
    df = _load_df(path)
    if df.empty:
        return {"error": f"Brak danych w {path}"}
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict(orient="records")

def _render_html_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<html><body><h3>{title}</h3><p>Brak danych.</p></body></html>"

    # sort po confidence
    if "confidence_score" in df.columns:
        df = df.sort_values("confidence_score", ascending=False).copy()

    # zaokrąglenia
    num_cols = [
        "line","E_min","sd_min",
        "mu_baseline","p_over_baseline","fair_line_baseline",
        "mu_glm","p_over_glm","fair_line_glm",
        "mu_negbin","p_over_negbin","fair_line_negbin",
        "mu_consensus","sigma_consensus","p_over_consensus","fair_line_consensus",
        "edge_points","z_consensus","confidence_score","disagreement"
    ]
    for c in [c for c in num_cols if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    # dobór kolumn
    cols = [c for c in [
        "player_name","market","line",
        "E_min","sd_min",
        "mu_baseline","p_over_baseline","fair_line_baseline",
        "mu_glm","p_over_glm","fair_line_glm",
        "mu_negbin","p_over_negbin","fair_line_negbin",
        "mu_consensus","sigma_consensus","p_over_consensus","fair_line_consensus",
        "edge_points","z_consensus","recommendation","confidence_bucket","confidence_score",
        "disagreement"
    ] if c in df.columns]
    df = df[cols].copy()

    # kolorowa plakietka dla recommendation
    def badge(rec: str) -> str:
        rec = str(rec).upper()
        color = "#00b050" if rec == "OVER" else ("#d00000" if rec == "UNDER" else "#555")
        bg = "#e8fff0" if rec == "OVER" else ("#fff0f0" if rec == "UNDER" else "#f2f2f2")
        return f"<span style='background:{bg};color:{color};padding:2px 8px;border-radius:12px;font-weight:600'>{rec}</span>"

    if "recommendation" in df.columns:
        df["recommendation"] = df["recommendation"].apply(badge)

    table_html = df.to_html(index=False, escape=False)
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          body{{font-family:Segoe UI,Inter,Arial,sans-serif;padding:18px}}
          h2{{margin-top:0}}
          table{{border-collapse:collapse;width:100%;font-size:14px}}
          th,td{{border:1px solid #ddd;padding:6px 8px}}
          th{{background:#111;color:#fff;position:sticky;top:0}}
          tr:nth-child(even){{background:#fafafa}}
        </style>
      </head>
      <body>
        <h2>{title}</h2>
        {table_html}
      </body>
    </html>
    """

@app.get("/predict/today")
def predict_today():
    d = date.today()
    return _respond(run_day(d), ["player_name","market","line","mu","sigma","p10","p50","p90","p_over"])

@app.get("/predict/{date_str}")
def predict_date(date_str: str):
    d = date.fromisoformat(date_str)
    return _respond(run_day(d), ["player_name","market","line","mu","sigma","p10","p50","p90","p_over"])

@app.get("/props/compare/today")
def compare_today_json():
    d = date.today()
    cols = ["player_name","market","line",
            "E_min","sd_min",
            "mu_baseline","p_over_baseline","fair_line_baseline",
            "mu_glm","p_over_glm","fair_line_glm",
            "mu_negbin","p_over_negbin","fair_line_negbin",
            "mu_consensus","sigma_consensus","p_over_consensus","fair_line_consensus",
            "edge_points","z_consensus","recommendation","confidence_bucket","confidence_score",
            "disagreement"]
    return _respond(run_compare_day(d), cols)

@app.get("/props/compare/{date_str}")
def compare_date_json(date_str: str):
    d = date.fromisoformat(date_str)
    cols = ["player_name","market","line",
            "E_min","sd_min",
            "mu_baseline","p_over_baseline","fair_line_baseline",
            "mu_glm","p_over_glm","fair_line_glm",
            "mu_negbin","p_over_negbin","fair_line_negbin",
            "mu_consensus","sigma_consensus","p_over_consensus","fair_line_consensus",
            "edge_points","z_consensus","recommendation","confidence_bucket","confidence_score",
            "disagreement"]
    return _respond(run_compare_day(d), cols)

@app.get("/props/compare/table/today", response_class=HTMLResponse)
def compare_today_table():
    d = date.today()
    df = _load_df(run_compare_day(d))
    return _render_html_table(df, "NBA Props — Models Compare (today)")

@app.get("/props/compare/table/{date_str}", response_class=HTMLResponse)
def compare_date_table(date_str: str):
    d = date.fromisoformat(date_str)
    df = _load_df(run_compare_day(d))
    return _render_html_table(df, f"NBA Props — Models Compare ({d.isoformat()})")
