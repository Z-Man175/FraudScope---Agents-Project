import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from pattern_agent.scenarios import build_dataset

df = build_dataset()
X = pd.DataFrame({
    "log_amount": np.log1p(df["amount"]),
    "hour": df["step"] % 24,
    "is_transfer": (df["type"] == "TRANSFER").astype(int),
    "is_cash_out": (df["type"] == "CASH_OUT").astype(int),
    "is_payment": (df["type"] == "PAYMENT").astype(int),
})
# fraud can only live in TRANSFER and CASH_OUT (as in PaySim), so compare against ordinary
# rows of those two types, otherwise "it is not a PAYMENT" would give the injected rows away
normal = (df["kind"] == "normal") & df["type"].isin(["TRANSFER", "CASH_OUT"])

print("Can a classifier tell injected rows from ordinary TRANSFER/CASH_OUT rows using only type, amount, hour?")
print("(AUC 0.5 = cannot tell, 1.0 = trivial. Below ~0.75 is OK.)\n")
for scen in ["mule_burst", "small_repeated", "account_chain", "account_takeover",
             "payroll", "rent_subscription", "business_forwarding", "big_purchase"]:
    mask = normal | (df["scenario"] == scen)
    y = (df.loc[mask, "scenario"] == scen).astype(int)
    # keep every case (and every ordinary account) in ONE fold, so near-identical rows
    # of the same case never sit on both sides of the split
    groups = df.loc[mask].apply(lambda r: r["scenario_id"] or r["nameOrig"], axis=1)
    clf = GradientBoostingClassifier(random_state=42)
    p = cross_val_predict(clf, X[mask], y, groups=groups, cv=GroupKFold(5),
                          method="predict_proba")[:, 1]
    print(f"  {scen:22s} AUC {roc_auc_score(y, p):.2f}")