import pandas as pd
from pattern_agent.schema import prepare
from pattern_agent.detectors import detect_account_chains

COLS = ["step", "type", "amount", "nameOrig", "nameDest"]


def make_toy():
    rows = []
    # 1. 3 hops, each forwards 85-90%                      -> CHAIN (3 hops)
    rows += [(1, "TRANSFER", 1000, "C11", "C12"), (2, "TRANSFER", 900, "C12", "C13"), (3, "CASH_OUT", 850, "C13", "C14")]
    # 2. middle hop forwards only 30%                      -> no chain (ratio too low)
    rows += [(1, "TRANSFER", 1000, "C21", "C22"), (2, "TRANSFER", 300, "C22", "C23"), (3, "CASH_OUT", 290, "C23", "C24")]
    # 3. hops 10 and 20 steps apart                        -> no chain (too slow)
    rows += [(1, "TRANSFER", 1000, "C31", "C32"), (10, "TRANSFER", 900, "C32", "C33"), (20, "CASH_OUT", 850, "C33", "C34")]
    # 4. only 2 hops                                       -> no chain (too short)
    rows += [(1, "TRANSFER", 1000, "C41", "C42"), (2, "CASH_OUT", 900, "C42", "C43")]
    # 5. 5 hops                                            -> CHAIN (one flag with 5 hops)
    rows += [(1, "TRANSFER", 1000, "C51", "C52"), (2, "TRANSFER", 950, "C52", "C53"), (3, "TRANSFER", 900, "C53", "C54"),
             (4, "TRANSFER", 860, "C54", "C55"), (5, "CASH_OUT", 820, "C55", "C56")]
    # 6. money bouncing back and forth between two accounts -> no chain
    rows += [(1, "TRANSFER", 1000, "C61", "C62"), (2, "TRANSFER", 950, "C62", "C61"), (3, "TRANSFER", 900, "C61", "C62")]
    return prepare(pd.DataFrame(rows, columns=COLS))


if __name__ == "__main__":
    flags = detect_account_chains(make_toy())
    for f in flags:
        print(f.pattern, " -> ".join(f.accounts), "| hops:", f.details["hops"], "| strength:", f.strength)
    got = {f.accounts[0]: f.details["hops"] for f in flags}
    want = {"C11": 3, "C51": 5}
    print("PASS" if got == want else f"FAIL: got {got}, wanted {want}")