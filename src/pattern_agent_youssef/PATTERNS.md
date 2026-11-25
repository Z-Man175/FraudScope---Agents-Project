# FraudScope Pattern Agent: Pattern Definitions

Detectors use only `step, type, amount, nameOrig, nameDest` (no balance columns).
Time unit: 1 step = 1 hour. Split: train 1-480, val 481-600, test 601-743.

## Detectors

| Pattern | Rule |
|---|---|
| burst | 5+ transactions from OR to one customer account within 6 steps, and at least 3x that account's usual count per window (earlier activity only). Merchants skipped on the "to" side. |
| repeated_recipient | 4+ TRANSFER/CASH_OUT to one customer recipient within 24 steps, each below the sender's typical (median) amount. Typical ignores payments to the same recipient; peer median is used if the sender has fewer than 3 earlier transactions. |
| account_chain | 3+ hops within 12 steps; each hop forwards 80-100% of what the account just received; forward must come after receipt; A->B->A does not count. |

Flag format: flag_id, pattern, accounts, tx_ids, window_start, window_end, strength (0-1), details.

## Fraud scenarios

| Scenario | What is injected | Caught by |
|---|---|---|
| mule_burst | Dormant account funded by a victim, then 6-9 transfers to different people within 4 steps | burst |
| small_repeated | Established account sends 4-7 transfers to one new recipient within 20 steps, each 25-70% of its typical amount | repeated_recipient |
| account_chain (HELD OUT, test only) | 3-6 hops, each forwarding 82-98%, 1-2 steps apart | account_chain |
| account_takeover | One transfer 3.2-4.5x the victim's past max to a new recipient, then a cash-out | none (single-transaction model, Ziad) |

## Look-alike legitimate cases

| Look-alike | Twin of | Why it looks fraudulent | How history clears it |
|---|---|---|---|
| payroll | mule_burst | One employer pays 8-14 people within 2 steps | Same staff, same weekday/hour, same fixed pay every week |
| rent_subscription | small_repeated | Many small transfers to one recipient | Same amount every 8 steps to a known payee |
| business_forwarding | account_chain | Funds hop through 4 accounts at 86-95% | Same counterparties and stable margin, repeats weekly |
| big_purchase | account_takeover | Very large transfer to a new recipient | Account made large transfers before |


## Held-out pattern
account_chain (fraud and business_forwarding look-alikes) is marked `held_out=True` in
`scenario_labels.csv`. Fraud chains exist only in the test period.

## Known limitations
- Synthetic data only; not yet run on real PaySim.
- Recall of 100% on injected fraud is by construction, not a result.
- Rent is modelled as every 8 steps (simulation is ~31 days).
- Strength is count-based, so large payroll series can outrank fraud at small K_p.