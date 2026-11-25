import numpy as np
import pandas as pd

SEED = 42
N_STEPS = 743
TRAIN_END, VAL_END = 480, 600      # train 1-480, validation 481-600, test 601-743
COLS = ["step", "type", "amount", "nameOrig", "nameDest"]

# daily rhythm: quiet at night, busy in the afternoon and evening
HOUR_WEIGHTS = np.array([0.3] * 6 + [1.0] * 12 + [1.4] * 4 + [0.6] * 2)


def split_of(step):
    if step <= TRAIN_END:
        return "train"
    if step <= VAL_END:
        return "val"
    return "test"


def make_background(n_accounts=400, n_merchants=60, seed=SEED):
    """Ordinary customers. Each has its own typical amount, activity level
    and a few usual contacts, so 'normal' differs from account to account."""
    rng = np.random.default_rng(seed)
    accounts = [f"C{100000 + i}" for i in range(n_accounts)]
    merchants = [f"M{200000 + i}" for i in range(n_merchants)]

    all_steps = np.arange(1, N_STEPS + 1)
    step_p = np.array([HOUR_WEIGHTS[s % 24] for s in all_steps])
    step_p = step_p / step_p.sum()

    rows = []
    for acct in accounts:
        typical = rng.lognormal(mean=8.0, sigma=0.7)        # this account's usual amount
        n_tx = max(3, int(rng.lognormal(mean=3.4, sigma=0.5)))
        contacts = [a for a in rng.choice(accounts, size=5, replace=False) if a != acct][:4]
        steps = rng.choice(all_steps, size=n_tx, p=step_p)

        for s in steps:
            kind = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"],
                              p=[0.45, 0.30, 0.20, 0.05])
            amount = round(float(typical * rng.lognormal(0, 0.6)), 2)
            if kind == "PAYMENT":
                dest = str(rng.choice(merchants))
            elif kind == "TRANSFER" and rng.random() < 0.65:
                dest = str(rng.choice(contacts))             # usual contact
            else:
                dest = str(rng.choice(accounts))
                while dest == acct:
                    dest = str(rng.choice(accounts))
            rows.append((int(s), kind, amount, acct, dest))

    df = pd.DataFrame(rows, columns=COLS)
    df["is_fraud"] = 0
    df["kind"] = "normal"          # normal | fraud | lookalike
    df["scenario"] = "normal"      # which scenario made this row
    df["scenario_id"] = ""         # which case, e.g. "mule_burst-003"
    return df


# ---------------------------------------------------------------------------
# Step 5b: fraud scenarios
# ---------------------------------------------------------------------------
STEP_RANGE = {"train": (1, TRAIN_END), "val": (TRAIN_END + 1, VAL_END), "test": (VAL_END + 1, N_STEPS)}
WARMUP = 200       # no scenario starts before this step, so accounts already have some history

# how many cases of each pattern go into each period
FRAUD_COUNTS = {
    "mule_burst":       {"train": 10, "val": 4, "test": 6},
    "small_repeated":   {"train": 10, "val": 4, "test": 6},
    "account_chain":    {"test": 15},                      # HELD OUT: test period only
    "account_takeover": {"train": 20, "val": 8, "test": 10},
}


def _range(split):
    lo, hi = STEP_RANGE[split]
    return max(lo, WARMUP), hi


def _start_step(rng, lo, hi, night=False):
    """Pick a start step, following the same daily rhythm as ordinary traffic."""
    steps = np.arange(lo, hi + 1)
    if night:
        w = np.array([1.0 if s % 24 < 6 else 0.05 for s in steps])
    else:
        w = np.array([HOUR_WEIGHTS[s % 24] for s in steps])
    return int(rng.choice(steps, p=w / w.sum()))


def _row(step, typ, amount, orig, dest, scenario, sid, kind="fraud"):
    return dict(step=int(step), type=typ, amount=round(float(amount), 2),
                nameOrig=orig, nameDest=dest, is_fraud=1 if kind == "fraud" else 0,
                kind=kind, scenario=scenario, scenario_id=sid)


def _pick_account(bg, rng, before, min_prior, max_prior, used):
    """A background account with between min_prior and max_prior transactions
    BEFORE step `before`, not already used by another scenario."""
    counts = bg[bg["step"] < before].groupby("nameOrig").size()
    ok = counts[(counts >= min_prior) & (counts <= max_prior)].index
    ok = [a for a in ok if a not in used]
    acct = str(rng.choice(ok))
    used.add(acct)
    return acct


def _pick_typical(bg, rng, before, lo_mult, hi_mult, used, min_prior=8):
    """Like _pick_account, but only accounts whose usual amount is between lo_mult and
    hi_mult times the usual amount of everyone. Keeps injected amounts in the same
    range as ordinary ones, so the amount alone does not give a case away."""
    stats = bg[bg["step"] < before].groupby("nameOrig")["amount"].agg(["size", "median"])
    ref = bg["amount"].median()
    ok = stats[(stats["size"] >= min_prior) & (stats["median"] >= lo_mult * ref)
               & (stats["median"] <= hi_mult * ref)].index
    ok = [a for a in ok if a not in used]
    acct = str(rng.choice(ok))
    used.add(acct)
    return acct


def _other_accounts(bg, rng, n, exclude):
    pool = [a for a in bg["nameOrig"].unique() if a not in exclude]
    return [str(a) for a in rng.choice(pool, size=n, replace=False)]


def _prior(bg, acct, before):
    return bg[(bg["nameOrig"] == acct) & (bg["step"] < before)]


def _mule_burst(bg, rng, split, sid, used):
    """A dormant account is funded, then quickly sends many transfers to different people."""
    lo, hi = _range(split)
    start = _start_step(rng, lo + 3, hi - 6)
    mule = _pick_account(bg, rng, start, 3, 20, used)
    victim = _pick_account(bg, rng, start, 5, 10_000, used)
    n = int(rng.integers(6, 10))
    recips = _other_accounts(bg, rng, n, {mule, victim})
    parts = bg["amount"].median() * rng.lognormal(0, 0.9, n)
    steps = start + np.sort(rng.integers(0, 5, n))                 # all within 4 steps

    rows = [_row(start - int(rng.integers(1, 4)), "TRANSFER", parts.sum() * 1.02,
                 victim, mule, "mule_burst", sid)]                  # the funding
    for s, amt, r in zip(steps, parts, recips):
        typ = "TRANSFER" if rng.random() < 0.7 else "CASH_OUT"
        rows.append(_row(s, typ, amt, mule, r, "mule_burst", sid))
    return rows


def _small_repeated(bg, rng, split, sid, used):
    """An established account sends several small transfers to one new recipient."""
    lo, hi = _range(split)
    start = _start_step(rng, lo + 30, hi - 24)
    sender = _pick_typical(bg, rng, start, 1.2, 3.0, used)
    past = _prior(bg, sender, start)
    recipient = _other_accounts(bg, rng, 1, set(past["nameDest"]) | {sender})[0]
    typical = past["amount"].median()
    n = int(rng.integers(4, 8))
    window = np.arange(start, start + 21)                          # all within 20 steps
    w = np.array([HOUR_WEIGHTS[x % 24] for x in window])
    steps = np.sort(rng.choice(window, size=n, p=w / w.sum()))     # busier hours more likely
    amounts = typical * rng.uniform(0.25, 0.7, n)                  # each below typical
    return [_row(s, "TRANSFER" if rng.random() < 0.6 else "CASH_OUT", a, sender, recipient,
                 "small_repeated", sid) for s, a in zip(steps, amounts)]


def _account_chain(bg, rng, split, sid, used):
    """Money hops through successive accounts, each forwarding 82-98% of what it got."""
    lo, hi = _range(split)
    start = _start_step(rng, lo, hi - 12)
    hops = int(rng.integers(3, 7))
    first = _pick_typical(bg, rng, start, 0.3, 1.0, used, min_prior=5)
    others = _other_accounts(bg, rng, hops, used | {first})
    used.update(others)
    path = [first] + others                                        # hops + 1 accounts
    typical = _prior(bg, first, start)["amount"].median()
    amount = typical * rng.uniform(1.5, 4.0)
    step = start
    rows = []
    for h in range(hops):
        typ = "CASH_OUT" if h == hops - 1 else "TRANSFER"
        rows.append(_row(step, typ, amount, path[h], path[h + 1], "account_chain", sid))
        amount *= rng.uniform(0.82, 0.98)
        step += int(rng.integers(1, 3))                            # 1-2 steps between hops
    return rows


def _account_takeover(bg, rng, split, sid, used):
    """One very large transfer out of an established account to a new recipient, then cash-out."""
    lo, hi = _range(split)
    start = _start_step(rng, lo + 30, hi - 4)
    victim = _pick_typical(bg, rng, start, 0.15, 0.6, used)
    past = _prior(bg, victim, start)
    recipient = _other_accounts(bg, rng, 1, set(past["nameDest"]) | {victim})[0]
    agent = _other_accounts(bg, rng, 1, {victim, recipient})[0]
    amount = past["amount"].max() * rng.uniform(3.2, 4.5)          # well above anything it did before
    return [
        _row(start, "TRANSFER", amount, victim, recipient, "account_takeover", sid),
        _row(start + int(rng.integers(1, 4)), "CASH_OUT", amount * rng.uniform(0.95, 1.0),
             recipient, agent, "account_takeover", sid),
    ]


FRAUD_MAKERS = {"mule_burst": _mule_burst, "small_repeated": _small_repeated,
                "account_chain": _account_chain, "account_takeover": _account_takeover}


def inject_fraud(bg, seed=SEED):
    """Return the background plus every fraud case, labelled."""
    rng = np.random.default_rng(seed + 1)
    used, new_rows = set(), []
    for scenario, per_split in FRAUD_COUNTS.items():
        for split, k in per_split.items():
            for i in range(k):
                sid = f"{scenario}-{split}-{i:03d}"
                new_rows += FRAUD_MAKERS[scenario](bg, rng, split, sid, used)
    return pd.concat([bg, pd.DataFrame(new_rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 5c: look-alike legitimate cases, labels, held-out rule, export
# ---------------------------------------------------------------------------
WEEK = 168   # steps in a week

# payroll, rent_subscription and business_forwarding are RECURRING series (they need history
# to look legitimate), so they run across the whole timeline: "all".
LOOKALIKE_COUNTS = {
    "payroll":             {"all": 4},
    "rent_subscription":   {"all": 6},
    "business_forwarding": {"all": 3},
    "big_purchase":        {"train": 15, "val": 6, "test": 8},
}

# which detector family each scenario belongs to (look-alikes share their fraud twin's family)
SCENARIO_PATTERN = {
    "mule_burst": "burst",                 "payroll": "burst",
    "small_repeated": "repeated_recipient", "rent_subscription": "repeated_recipient",
    "account_chain": "account_chain",       "business_forwarding": "account_chain",
    "account_takeover": "single_transaction", "big_purchase": "single_transaction",
}
HELD_OUT = "account_chain"     # fraud AND look-alike rows of this family are excluded from memory/training


def _idx(sid):
    return int(sid.split("-")[-1])


def _payroll(bg, rng, split, sid, used):
    """One employer pays the same 8-14 staff on the same weekday and hour, every week."""
    employer = _pick_account(bg, rng, 10_000, 0, 10_000, used)
    n = int(rng.integers(8, 15))
    staff = _other_accounts(bg, rng, n, used | {employer})
    salary = bg["amount"].median() * rng.lognormal(0, 0.9, n)       # each person's fixed pay
    s0 = 12 * _idx(sid) + int(rng.integers(8, 18))                  # same weekday and hour each week
    rows = []
    for week in range(5):
        base = s0 + WEEK * week
        steps = base + np.sort(rng.integers(0, 3, n))               # all within 2 steps
        for s, a, person in zip(steps, salary, staff):
            rows.append(_row(s, "TRANSFER", a * rng.uniform(0.99, 1.01), employer, person,
                             "payroll", sid, kind="lookalike"))
    return rows


def _rent_subscription(bg, rng, split, sid, used):
    """A fixed small payment to the same known recipient at a regular interval."""
    s0 = 120 + 90 * _idx(sid)
    payer = _pick_typical(bg, rng, s0, 1.0, 3.0, used)
    past = _prior(bg, payer, s0)
    payee = _other_accounts(bg, rng, 1, {payer})[0]
    amount = past["amount"].median() * rng.uniform(0.25, 0.45)      # always below its typical
    n = min(20, (N_STEPS - s0) // 8)
    return [_row(s0 + 8 * k, "TRANSFER", amount * rng.uniform(0.99, 1.01), payer, payee,
                 "rent_subscription", sid, kind="lookalike") for k in range(n)]


def _business_forwarding(bg, rng, split, sid, used):
    """A trade chain (shop -> distributor -> wholesaler -> factory) that repeats every week
    with the same counterparties and a stable margin at each hop."""
    path = _other_accounts(bg, rng, 4, used)
    used.update(path)
    margins = rng.uniform(0.86, 0.95, 3)                           # stable per series
    gap = int(rng.integers(1, 3))
    base_amount = bg["amount"].median() * rng.lognormal(0.3, 0.8)
    s0 = 12 * _idx(sid) + int(rng.integers(8, 18))
    rows = []
    for week in range(5):
        amount = base_amount * rng.uniform(0.9, 1.1)                # volume varies, margin does not
        step = s0 + WEEK * week
        for h in range(3):
            rows.append(_row(step, "TRANSFER", amount, path[h], path[h + 1],
                             "business_forwarding", sid, kind="lookalike"))
            amount *= margins[h] * rng.uniform(0.995, 1.005)
            step += gap
    return rows


def _big_purchase(bg, rng, split, sid, used):
    """An account that has made large transfers before makes another one."""
    lo, hi = _range(split)
    start = _start_step(rng, lo + 30, hi)
    acct = _pick_typical(bg, rng, start, 0.3, 0.7, used)
    past = _prior(bg, acct, start)
    typical = past["amount"].median()
    recips = _other_accounts(bg, rng, 3, set(past["nameDest"]) | {acct})
    earlier = [_start_step(rng, 5, start - 24) for _ in range(2)]     # its large transfers BEFORE
    rows = [_row(s, "TRANSFER", typical * rng.uniform(5, 10), acct, r, "big_purchase", sid, kind="lookalike")
            for s, r in zip(earlier, recips[:2])]
    rows.append(_row(start, "TRANSFER", typical * rng.uniform(5, 10), acct, recips[2],
                     "big_purchase", sid, kind="lookalike"))
    return rows


LOOKALIKE_MAKERS = {"payroll": _payroll, "rent_subscription": _rent_subscription,
                    "business_forwarding": _business_forwarding, "big_purchase": _big_purchase}


def build_dataset(seed=SEED):
    """Background + fraud + look-alikes, labelled and ready to use."""
    from .schema import prepare
    bg = make_background(seed=seed)
    rng = np.random.default_rng(seed + 1)
    used, rows = set(), []
    for counts, makers in ((FRAUD_COUNTS, FRAUD_MAKERS), (LOOKALIKE_COUNTS, LOOKALIKE_MAKERS)):
        for scenario, per_split in counts.items():
            for split, k in per_split.items():
                for i in range(k):
                    sid = f"{scenario}-{split}-{i:03d}"
                    rows += makers[scenario](bg, rng, split, sid, used)

    df = prepare(pd.concat([bg, pd.DataFrame(rows)], ignore_index=True))
    df["split"] = df["step"].map(split_of)
    df["pattern"] = df["scenario"].map(SCENARIO_PATTERN).fillna("none")
    df["held_out"] = df["pattern"] == HELD_OUT
    return df


def export(df, outdir="data"):
    """transactions.csv for Ziad (no scenario info), scenario_labels.csv for Jana and Freedy."""
    import os
    os.makedirs(outdir, exist_ok=True)
    df[["tx_id", "step", "type", "amount", "nameOrig", "nameDest", "is_fraud"]].rename(
        columns={"is_fraud": "isFraud"}).to_csv(f"{outdir}/transactions.csv", index=False)
    df[["tx_id", "is_fraud", "kind", "scenario", "pattern", "scenario_id", "split", "held_out"]].to_csv(
        f"{outdir}/scenario_labels.csv", index=False)