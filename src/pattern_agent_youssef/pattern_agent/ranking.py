import pandas as pd
from .detectors import detect_bursts, detect_repeated_recipients, detect_account_chains


def run_detectors(df):
    """Run the three pattern detectors and give every flag an id like 'burst-00001'."""
    flags = detect_bursts(df) + detect_repeated_recipients(df) + detect_account_chains(df)
    counters = {}
    for f in flags:
        counters[f.pattern] = counters.get(f.pattern, 0) + 1
        f.flag_id = f"{f.pattern}-{counters[f.pattern]:05d}"
    return flags


def placeholder_anomaly(df):
    """STAND-IN for Ziad's anomaly score, so the ranking can be built and tested now.
    Score = how big the amount is compared with the account's own past median (percentile).
    Same shape Ziad will deliver: a Series indexed by tx_id, higher = more suspicious."""
    prior_median = df.groupby("nameOrig")["amount"].transform(
        lambda s: s.expanding().median().shift(1))
    ratio = (df["amount"] / prior_median).fillna(1.0)
    return pd.Series(ratio.rank(pct=True).to_numpy(), index=df["tx_id"].to_numpy())


def rank_candidates(flags, anomaly, k_a=20, k_p=20, pattern_weight=0.5):
    """Quota ranking: the top k_a transactions by anomaly score PLUS the top k_p pattern flags
    by strength, deduplicated. One candidate = one transaction (anomaly) or one flag (pattern)."""
    # anomaly side: single transactions
    top = anomaly.nlargest(k_a) if k_a > 0 else anomaly.iloc[0:0]
    anom = [{"source": "anomaly", "tx_ids": [int(t)], "anomaly_score": float(s),
             "pattern_strength": 0.0, "flag_ids": []} for t, s in top.items()]

    # pattern side: whole cases. Strongest first; ties go to the larger, then the earlier case.
    best = sorted(flags, key=lambda f: (-f.strength, -len(f.tx_ids), f.window_start))[:k_p]
    pat = [{"source": "pattern", "tx_ids": [int(t) for t in f.tx_ids],
            "anomaly_score": float(anomaly.reindex(f.tx_ids).max()),
            "pattern_strength": float(f.strength), "flag_ids": [f.flag_id]} for f in best]

    # deduplicate: an anomaly candidate that sits inside a pattern case is merged into it
    where = {c["tx_ids"][0]: i for i, c in enumerate(anom)}
    absorbed = set()
    for p in pat:
        hits = [where[t] for t in p["tx_ids"] if t in where]
        if hits:
            p["source"] = "both"
            absorbed.update(hits)
    out = pat + [c for i, c in enumerate(anom) if i not in absorbed]

    for c in out:
        c["priority"] = round(pattern_weight * c["pattern_strength"]
                              + (1 - pattern_weight) * c["anomaly_score"], 4)
    out.sort(key=lambda c: -c["priority"])
    for i, c in enumerate(out, 1):
        c["candidate_id"] = f"cand-{i:04d}"
    return out


def cases_reached(candidates, df):
    """Which fraud cases have at least one transaction inside some candidate?"""
    in_list = {t for c in candidates for t in c["tx_ids"]}
    cases = df[df["is_fraud"] == 1].groupby("scenario_id")["tx_id"].apply(set)
    reached = {sid for sid, txs in cases.items() if txs & in_list}
    return reached, set(cases.index)