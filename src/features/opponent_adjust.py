from __future__ import annotations
import pandas as pd

def build_allowed_factors(gamelogs: pd.DataFrame,
                          window_games: int = 10,
                          required_team_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Team-level 'allowed' PTS/REB/AST vs średnia ligi.
    Gdy brak gamelogów – zwraca czynniki = 1.0 dla przekazanych team_id (jeśli podane),
    albo pustą ramkę (model wtedy pominie korektę).
    """
    if gamelogs is None or gamelogs.empty:
        if required_team_ids:
            return pd.DataFrame({
                "team_id": required_team_ids,
                "f_pts": 1.0, "f_reb": 1.0, "f_ast": 1.0
            })
        return pd.DataFrame(columns=["team_id","f_pts","f_reb","f_ast"])

    keep_cols = ["game_id", "team_id", "game_date", "pts", "reb", "ast"]
    tg = gamelogs[keep_cols].groupby(["game_id", "team_id"], as_index=False).agg(
        pts=("pts", "sum"),
        reb=("reb", "sum"),
        ast=("ast", "sum"),
        game_date=("game_date", "max"),
    )

    pairs = tg.merge(tg, on="game_id", suffixes=("_def", "_opp"))
    pairs = pairs[pairs["team_id_def"] != pairs["team_id_opp"]].copy()
    pairs = pairs[["team_id_def", "game_date_def", "pts_opp", "reb_opp", "ast_opp"]].rename(
        columns={"team_id_def":"team_id","game_date_def":"game_date",
                 "pts_opp":"allowed_pts","reb_opp":"allowed_reb","ast_opp":"allowed_ast"}
    ).sort_values(["team_id","game_date"])

    def take_tail_mean(df):
        tail = df.tail(window_games)
        return pd.Series({
            "allowed_pts": tail["allowed_pts"].mean(),
            "allowed_reb": tail["allowed_reb"].mean(),
            "allowed_ast": tail["allowed_ast"].mean(),
        })

    recent = pairs.groupby("team_id", as_index=False).apply(take_tail_mean)
    league = recent[["allowed_pts","allowed_reb","allowed_ast"]].mean()

    recent["f_pts"] = (recent["allowed_pts"]/league["allowed_pts"]).clip(0.8, 1.2)
    recent["f_reb"] = (recent["allowed_reb"]/league["allowed_reb"]).clip(0.8, 1.2)
    recent["f_ast"] = (recent["allowed_ast"]/league["allowed_ast"]).clip(0.8, 1.2)

    return recent[["team_id","f_pts","f_reb","f_ast"]]
