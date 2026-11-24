"""Build the boilerplate list: sentences that appear (after normalising) in >= 3 different TRAIN companies.

Such sentences are standard wording (accounting-policy text, statutory headers, sensitivity-analysis boilerplate),
not company-specific events. Only train companies are used, so no test/val/backtest text leaks in.
  python src/models/build_boilerplate.py   ->  models/boilerplate.txt
"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from rules import norm  # noqa: E402

HELD_OUT = {"JETAIRWAYS", "YESBANK", "TCS", "MARUTI", "SUNPHARMA", "RCOM", "TITAN", "NTPC", "GENSOL"}
MIN_COMPANIES = 3

seen = defaultdict(set)
with gzip.open(ROOT / "data/interim/sentences.jsonl.gz", "rt") as f:
    for line in f:
        s = json.loads(line)
        if s["company"] not in HELD_OUT:
            k = norm(s["text"])
            if len(k.split()) >= 5:
                seen[k].add(s["company"])
keep = sorted(k for k, c in seen.items() if len(c) >= MIN_COMPANIES)
(ROOT / "models/boilerplate.txt").write_text("\n".join(keep) + "\n")
print(f"{len(seen)} distinct sentences from {len({c for v in seen.values() for c in v})} train companies; "
      f"{len(keep)} appear in >= {MIN_COMPANIES} companies -> models/boilerplate.txt")
