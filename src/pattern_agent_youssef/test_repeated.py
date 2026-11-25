import pandas as pd
from pattern_agent.schema import prepare
from pattern_agent.detectors import detect_repeated_recipients

COLS = ["step", "type", "amount", "nameOrig", "nameDest"]


def history(acct, tag):
    # 10 normal transfers of 1000 to 10 different people -> this account's "typical" is 1000
    return [(i * 3, "TRANSFER", 1000, acct, f"C{tag}{i}") for i in range(10)]


def make_toy():
    rows = []
    # C1: usual 1000, then 5 small (200) transfers to C999 in 16 steps  -> FLAG
    rows += history("C1", 10)
    rows += [(100 + i * 4, "TRANSFER", 200, "C1", "C999") for i in range(5)]
    # C2: usual 1000, then 5 BIG (5000) transfers to C998               -> no flag (not small)
    rows += history("C2", 20)
    rows += [(100 + i * 4, "TRANSFER", 5000, "C2", "C998") for i in range(5)]
    # C3: no history, 4 tiny (50) transfers to C997 in 12 steps         -> FLAG (peer median)
    rows += [(200 + i * 4, "TRANSFER", 50, "C3", "C997") for i in range(4)]
    # C4: only 3 small transfers to C996                                -> no flag (too few)
    rows += [(300 + i * 4, "TRANSFER", 200, "C4", "C996") for i in range(3)]
    # C5: 5 small transfers spread over 120 steps                       -> no flag (too slow)
    rows += [(400 + i * 30, "TRANSFER", 200, "C5", "C995") for i in range(5)]
    return prepare(pd.DataFrame(rows, columns=COLS))


if __name__ == "__main__":
    flags = detect_repeated_recipients(make_toy())
    for f in flags:
        print(f.pattern, f.accounts[0], "->", f.accounts[1], len(f.tx_ids), f.strength)

    got = {(f.accounts[0], f.accounts[1]) for f in flags}
    want = {("C1", "C999"), ("C3", "C997")}
    print("PASS" if got == want else f"FAIL: got {got}, wanted {want}")

    