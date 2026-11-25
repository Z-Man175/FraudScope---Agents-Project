# FraudScope: Pattern Analysis Agent (Youssef)

## Run
        cd src/pattern_agent_youssef
    pip install -r requirements.txt
    python run_all.py
## Tests
    python test_toy.py
    python test_repeated.py
    python test_chain.py
    python test_fraud.py
    python test_lookalikes.py
    python test_triviality.py
    python test_ranking.py

## Handoffs
| File | Goes to |
|---|---|
| outputs/flags.jsonl | Rina |
| data/scenario_labels.csv | Jana, Freedy |
| data/transactions.csv | Ziad |
| pattern_agent/ranking.py | Ziad (swap `placeholder_anomaly` for his scores) |

See PATTERNS.md for definitions.