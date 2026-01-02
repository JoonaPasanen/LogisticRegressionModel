import pandas as pd
import numpy as np

def get_data(start, end):
    dfs = []
    for year in range(start, end + 1):
        dfs.append(pd.read_csv(f"./data/mens/atp/atp_matches_{year}.csv"))
        try:
            if year == end:
                continue
            dfs.append(pd.read_csv(f"./data/mens/quali/atp_matches_qual_chall_{year}.csv"))
            dfs.append(pd.read_csv(f"./data/mens/futures/atp_matches_futures_{year}.csv"))
        except FileNotFoundError: 
            continue
        
    df = pd.concat(dfs, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values("tourney_date")
    return df

def mask_data(df, seed):
    np.random.seed(seed)
    mask = np.random.rand(len(df)) > 0.5
    df["player_a"] = np.where(mask, df["winner_name"], df["loser_name"])
    df["player_b"] = np.where(mask, df["loser_name"], df["winner_name"])
    df["player_a_rank"] = np.where(mask, df["winner_rank"], df["loser_rank"])
    df["player_b_rank"] = np.where(mask, df["loser_rank"], df["winner_rank"])
    df["player_a_points"] = np.where(mask, df["winner_rank_points"], df["loser_rank_points"])
    df["player_b_points"] = np.where(mask, df["loser_rank_points"], df["winner_rank_points"])
    df["player_a_age"] = np.where(mask, df["winner_age"], df["loser_age"])
    df["player_b_age"] = np.where(mask, df["loser_age"], df["winner_age"])
    df["winner"] = np.where(mask, 1, 0)
    return df

def compute_covariates(df):
    elos={}
    all_a = []
    surface_a = []
    all_b = []
    surface_b = []

    players = pd.concat([df["player_a"], df["player_b"]]).unique()

    for p in players:
        # All, Grass, Clay, Hard
        elos[p] = [1500, 1500, 1500, 1500]

    def update_elos(a, b, k, surface, winner):
        K = k
        Sa = winner
        Ea = 1 / (1 + 10**(-1*(elos[a][surface]-elos[b][surface])/400))
        Sb = not winner
        Eb = 1 / (1 + 10**(-1*(elos[b][surface]-elos[a][surface])/400))
        elos[a][surface] = elos[a][surface] + K*(Sa - Ea)
        elos[b][surface] = elos[b][surface] + K*(Sb - Eb)
        return


    for match in df.itertuples():
        a = match.player_a
        all_a.append(elos[a][0])
        b = match.player_b
        all_b.append(elos[b][0])
        update_elos(a, b, 16, 0, match.winner)

        if match.surface == "Grass":
            surface_a.append(elos[a][1])
            surface_b.append(elos[b][1])
            update_elos(a, b, 24, 1, match.winner)
        elif match.surface == "Clay":
            surface_a.append(elos[a][2])
            surface_b.append(elos[b][2])
            update_elos(a, b, 24, 2, match.winner)
        else:
            surface_a.append(elos[a][3])
            surface_b.append(elos[b][3])
            update_elos(a, b, 24, 3, match.winner)

    df["elo_a"] = all_a
    df["elo_b"] = all_b
    df["surface_elo_a"] = all_a
    df["surface_elo_b"] = all_b
    
    df["elo_diff"] = df["elo_a"] - df["elo_b"]
    df["surface_elo_diff"] = df["surface_elo_a"] - df["surface_elo_b"]

    df["rank_diff"] = df["player_b_rank"] - df["player_a_rank"]
    df["rank_points_diff"] = df["player_a_points"] - df["player_b_points"]
    df["age_diff"] = df["player_a_age"] - df["player_b_age"]


    keep_columns = [
        "tourney_date",
        "surface", 
        "rank_diff",
        "rank_points_diff", 
        "age_diff",
        "elo_diff",
        "surface_elo_diff",
        "winner"
    ]

    # Clean up the data
    df = df[keep_columns]
    df = df.dropna()
    dummies = pd.get_dummies(df["surface"], drop_first=True).astype("int")
    df = pd.concat([df, dummies], axis="columns")

    return df