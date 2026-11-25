from dataclasses import dataclass, asdict, field
import json
import pandas as pd

REQUIRED_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest"]


@dataclass
class Flag:
    flag_id: str         # e.g. "burst-00001"
    pattern: str         # "burst" | "repeated_recipient" | "account_chain"
    accounts: list       # accounts involved
    tx_ids: list         # transactions involved
    window_start: int    # first step
    window_end: int      # last step
    strength: float      # 0 to 1
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def prepare(df):
    """Check columns, add tx_id if missing, sort by time."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    if "tx_id" not in df.columns:
        df.insert(0, "tx_id", range(len(df)))
    return df.sort_values(["step", "tx_id"], kind="stable").reset_index(drop=True)


def save_flags(flags, path):
    with open(path, "w") as f:
        for fl in flags:
            f.write(json.dumps(fl.to_dict()) + "\n")