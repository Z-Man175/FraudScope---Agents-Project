from .schema import Flag


def dense_episodes(steps, min_count, window):
    """steps = sorted list of step numbers.
    Returns [(a, b)] index ranges that hold >= min_count events within `window` steps."""
    intervals, left = [], 0
    for right in range(len(steps)):
        while steps[right] - steps[left] > window:
            left += 1
        if right - left + 1 >= min_count:
            intervals.append((left, right))
    merged = []
    for l, r in intervals:
        if merged and l <= merged[-1][1]:       # overlaps the previous one -> merge
            merged[-1][1] = max(merged[-1][1], r)
        else:
            merged.append([l, r])
    return [(a, b) for a, b in merged]


def detect_bursts(df, min_txns=5, window=6, rate_multiplier=3.0):
    """Burst = min_txns+ transactions from (or to) one account within `window` steps,
    AND at least `rate_multiplier` x what that account usually does in a window."""
    flags = []
    for direction in ("out", "in"):
        if direction == "out":
            key, other = "nameOrig", "nameDest"
            sub = df
        else:
            key, other = "nameDest", "nameOrig"
            sub = df[df["nameDest"].str.startswith("C")]   # skip merchants (names start with M)

        for acct, g in sub.groupby(key):
            if len(g) < min_txns:
                continue
            steps = g["step"].to_numpy()
            for a, b in dense_episodes(steps, min_txns, window):
                seg = g.iloc[a:b + 1]
                n = len(seg)

                # "above its usual rate": use only the account's EARLIER activity
                start = steps[a]
                n_prior = int((steps < start).sum())
                span = max(start - steps[0], window)
                usual = n_prior / span * window            # usual count per window
                ratio = n / max(usual, 1.0)
                if ratio < rate_multiplier:
                    continue

                flags.append(Flag(
                    flag_id="",
                    pattern="burst",
                    accounts=[acct] + seg[other].unique().tolist(),
                    tx_ids=seg["tx_id"].tolist(),
                    window_start=int(seg["step"].iloc[0]),
                    window_end=int(seg["step"].iloc[-1]),
                    strength=round(min(1.0, 0.5 + 0.1 * (n - min_txns)), 2),
                    details={"direction": direction, "txn_count": n,
                             "usual_count_in_window": round(float(usual), 2),
                             "rate_ratio": round(float(ratio), 2)},
                ))
    return flags

def detect_repeated_recipients(df, min_transfers=4, window=24, small_ratio=1.0,
                               min_history=3, types=("TRANSFER", "CASH_OUT")):
    """Many SMALL transfers from one account to ONE recipient within `window` steps.
    'Small' = below small_ratio x the sender's typical (median) amount."""
    flags = []
    cand = df[df["type"].isin(types) & df["nameDest"].str.startswith("C")]
    if cand.empty:
        return flags
    peer_typical = float(cand["amount"].median())      # fallback for senders with no history

    for (orig, dest), g in cand.groupby(["nameOrig", "nameDest"]):
        if len(g) < min_transfers:
            continue
        history = df[df["nameOrig"] == orig]
        for a, b in dense_episodes(g["step"].to_numpy(), min_transfers, window):
            seg = g.iloc[a:b + 1]
            start = seg["step"].iloc[0]

            # typical amount: only the PAST, and ignoring payments to this same recipient
            prior = history[(history["step"] < start) & (history["nameDest"] != dest)]["amount"]
            if len(prior) >= min_history:
                typical, source = float(prior.median()), "own_history"
            else:
                typical, source = peer_typical, "peer_median"

            # keep only the small transfers, then check the count and window again
            small = seg[seg["amount"] < small_ratio * typical]
            for c, d in dense_episodes(small["step"].to_numpy(), min_transfers, window):
                s = small.iloc[c:d + 1]
                n = len(s)
                flags.append(Flag(
                    flag_id="",
                    pattern="repeated_recipient",
                    accounts=[orig, dest],
                    tx_ids=s["tx_id"].tolist(),
                    window_start=int(s["step"].iloc[0]),
                    window_end=int(s["step"].iloc[-1]),
                    strength=round(min(1.0, 0.5 + 0.1 * (n - min_transfers)), 2),
                    details={"recipient": dest, "transfers": n,
                             "total_amount": round(float(s["amount"].sum()), 2),
                             "typical_amount": round(typical, 2),
                             "typical_source": source},
                ))
    return flags

def detect_account_chains(df, min_hops=3, window=12, fwd_min=0.8, fwd_max=1.0):
    """Chain = min_hops+ linked transactions (A->B, B->C, C->D ...) within `window` steps,
    where each hop forwards between fwd_min and fwd_max of what the account just received."""
    cols = ["tx_id", "step", "nameOrig", "nameDest", "amount"]
    first = df[cols].assign(pos=df.index)
    first = first[first["nameDest"].str.startswith("C") & (first["amount"] > 0)]
    second = df[cols].assign(pos=df.index)

    # every pair (tx that sends money TO an account, tx that sends money FROM that account)
    links = first.merge(second, left_on="nameDest", right_on="nameOrig", suffixes=("_a", "_b"))
    links = links[(links["pos_b"] > links["pos_a"])                    # forward comes later
                  & (links["step_b"] - links["step_a"] <= window)
                  & (links["nameDest_b"] != links["nameOrig_a"])]      # no A->B->A
    ratio = links["amount_b"] / links["amount_a"]
    links = links[(ratio >= fwd_min) & (ratio <= fwd_max + 1e-9)]
    if links.empty:
        return []

    # for each transaction: the longest chain that ENDS there
    depth, start, prev = {}, {}, {}
    for r in links.sort_values("pos_b").itertuples(index=False):
        d_a = depth.get(r.pos_a, 1)
        s_a = start.get(r.pos_a, r.step_a)
        if r.step_b - s_a > window:                  # whole chain must fit in the window
            continue
        if d_a + 1 > depth.get(r.pos_b, 1):
            depth[r.pos_b], start[r.pos_b], prev[r.pos_b] = d_a + 1, s_a, r.pos_a

    # turn the longest chains into flags, each transaction used only once
    flags, used = [], set()
    ends = [p for p, d in depth.items() if d >= min_hops]
    for pos in sorted(ends, key=lambda p: -depth[p]):
        path = [pos]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path.reverse()
        if used.intersection(path):
            continue
        used.update(path)

        rows = df.loc[path]
        amounts = rows["amount"].to_numpy()
        hops = len(path)
        flags.append(Flag(
            flag_id="",
            pattern="account_chain",
            accounts=[rows["nameOrig"].iloc[0]] + rows["nameDest"].tolist(),
            tx_ids=rows["tx_id"].tolist(),
            window_start=int(rows["step"].iloc[0]),
            window_end=int(rows["step"].iloc[-1]),
            strength=round(min(1.0, 0.6 + 0.1 * (hops - min_hops)), 2),
            details={"hops": hops,
                     "forward_ratios": [round(float(x), 3) for x in amounts[1:] / amounts[:-1]],
                     "amount_in": round(float(amounts[0]), 2),
                     "amount_out": round(float(amounts[-1]), 2)},
        ))
    return flags