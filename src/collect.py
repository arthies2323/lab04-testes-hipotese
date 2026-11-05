"""Coleta PRs mergeados de um repo publico via gh CLI e gera data/prs.csv.
Grupo 'com_bot' = PR com review/comentario de copilot-pull-request-reviewer.
Amostra 30 por grupo (semente fixa). Uso: python -m src.collect"""
import json, subprocess
import numpy as np, pandas as pd
from . import config

REPO = "dotnet/runtime"
BOT = "copilot-pull-request-reviewer"


def fetch(limit=600):
    out = subprocess.run(["gh", "pr", "list", "-R", REPO, "--state", "merged", "-L", str(limit),
                          "--json", "number,createdAt,mergedAt,reviews,comments"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out)


def build(prs):
    rows = []
    for p in prs:
        logins = {x["author"]["login"] for x in p["reviews"] + p["comments"] if x.get("author")}
        h = (pd.Timestamp(p["mergedAt"]) - pd.Timestamp(p["createdAt"])).total_seconds() / 3600
        if h <= 0:
            continue
        rows.append(dict(repo=REPO, pr=p["number"], grupo="com_bot" if BOT in logins else "sem_bot",
                         horas_ate_merge=round(h, 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build(fetch())
    print(df.grupo.value_counts())
    rng = np.random.default_rng(config.SEED)
    parts = [g.sample(min(len(g), config.N_PER_GROUP), random_state=int(rng.integers(1e9)))
             for _, g in df.groupby("grupo")]
    s = pd.concat(parts)
    s.sort_values(["grupo", "pr"]).to_csv("data/prs.csv", index=False)
    print(s.groupby("grupo").horas_ate_merge.describe())
