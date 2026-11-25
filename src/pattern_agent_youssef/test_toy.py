import pandas as pd
from pattern_agent.schema import prepare
from pattern_agent.detectors import detect_bursts

COLS = ["step", "type", "amount", "nameOrig", "nameDest"]


def make_toy():
    rows = []
    # 1. C1 sends 6 transfers in 3 steps                 -> BURST (out)
    rows += [(10 + i % 3, "TRANSFER", 100 + i, "C1", f"C{100 + i}") for i in range(6)]
    # 2. C2 sends 6 transfers, 10 steps apart            -> NOT a burst
    rows += [(i * 10, "TRANSFER", 100, "C2", f"C{200 + i}") for i in range(6)]
    # 3. Landlord C3 receives rent from 6 tenants        -> BURST (in)
    rows += [(20 + i % 3, "TRANSFER", 1000, f"C{300 + i}", "C3") for i in range(6)]
    # 4. Merchant M1 receives 6 payments                 -> NOT flagged (merchant)
    rows += [(30 + i % 3, "PAYMENT", 50, f"C{400 + i}", "M1") for i in range(6)]
    # 5. C5 is busy (a txn every 2 steps), then 5 in 3 steps -> NOT flagged (normal for C5)
    rows += [(i * 2, "TRANSFER", 100, "C5", f"C{500 + i}") for i in range(21)]
    rows += [(s, "TRANSFER", 100, "C5", f"C{600 + k}") for k, s in enumerate([50, 50, 51, 51, 52])]
    # 6. C6 has no history, then 5 in 3 steps            -> BURST (out)
    rows += [(s, "TRANSFER", 100, "C6", f"C{700 + k}") for k, s in enumerate([50, 50, 51, 51, 52])]
    return prepare(pd.DataFrame(rows, columns=COLS))


if __name__ == "__main__":
    df = make_toy()
    flags = detect_bursts(df)
    for f in flags:
        print(f.pattern, f.accounts[0], f.details["direction"], len(f.tx_ids), f.strength)

    got = {(f.accounts[0], f.details["direction"]) for f in flags}
    want = {("C1", "out"), ("C3", "in"), ("C6", "out")}
    print("PASS" if got == want else f"FAIL: got {got}, wanted {want}")