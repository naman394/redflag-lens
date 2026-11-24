"""Training-data augmentation (iteration v10), from TRAIN companies only (never val/test/backtest).

Adds two kinds of extra *non-risk* examples (a third, synthetic negation, was tried and dropped; see main()), aimed at the failure modes found in error analysis and the backtest:
  1. negated  - a train risk sentence with "not" inserted after its first auxiliary verb
                ("The Company has received a notice" -> "The Company has not received a notice").
                Deliberately a different construction from the robustness test, which prepends a CARO-style
                "there were no instances where ..." clause, so that test stays independent.
  2. nil      - real nil confirmations from unlabelled train-company text (is_nil_statement), e.g. CARO clauses.
  3. template - real questionnaire rows and boilerplate shared by >= 3 train companies (is_template / boilerplate).
  python src/models/augment.py  ->  data/labels/v1.0/train_aug.jsonl
"""
import gzip, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, ROOT, load  # noqa: E402
from rules import is_nil_statement, is_template, norm  # noqa: E402

random.seed(42)
AUX = re.compile(r"\b(has|have|had|is|are|was|were|will|may|could|would|shall|can)\b(?! not\b)", re.I)
HELD_OUT = {"JETAIRWAYS", "YESBANK", "TCS", "MARUTI", "SUNPHARMA", "RCOM", "TITAN", "NTPC", "GENSOL"}


EVENT = re.compile(r"\s+(been|received|defaulted|incurred|filed|issued|imposed|levied|initiated|delayed|identified|"
                   r"reported|served|admitted|passed|invoked|downgraded|qualified|suspended|resigned|declared|"
                   r"classified|attached|raised|made|subject|pending|under|in default)\b", re.I)


def negate(t: str):
    """Insert "not" after the first auxiliary, only when an event verb follows (so the meaning really flips)."""
    m = AUX.search(t)
    if not m or m.start() > 120 or not EVENT.match(t, m.end()):
        return None
    return t[:m.end()] + " not" + t[m.end():]


def row(base: dict, text: str, kind: str) -> dict:
    return {**{k: base.get(k) for k in ("company", "fy", "group", "page", "section", "split")},
            "sentence_id": f"{base['sentence_id']}_aug_{kind}", "text": text, "is_risk": 0, "category": None,
            "specificity": None, "aug": kind}


def main():
    tr = load("train")
    labelled = set(tr.sentence_id)
    assert not (set(tr.company) & HELD_OUT)
    out = []
    # Synthetic negation (negate() above) was tried and dropped: on inspection many rewrites stayed risky
    # ("has not defaulted ... and has incurred losses", "accounts have not been prepared on going concern basis"),
    # so labelling them non-risk would add label noise. Only real report text is used below.
    bp = set((ROOT / "models/boilerplate.txt").read_text().split("\n"))
    nil, tmpl = [], []
    with gzip.open(ROOT / "data/interim/sentences.jsonl.gz", "rt") as f:
        for line in f:
            s = json.loads(line)
            if s["company"] in HELD_OUT or s["sentence_id"] in labelled or s.get("is_fragment"):
                continue
            if is_nil_statement(s["text"]):
                nil.append(row(s, s["text"], "nil"))
            elif is_template(s["text"]) or norm(s["text"]) in bp:
                tmpl.append(row(s, s["text"], "template"))
    out += random.sample(nil, min(300, len(nil))) + random.sample(tmpl, min(300, len(tmpl)))
    with open(DATA / "train_aug.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(out), Counter(o["aug"] for o in out), f"(pool: nil {len(nil)}, template {len(tmpl)})")
    for o in out[:4]:
        print("  ", o["text"][:140])


if __name__ == "__main__":
    main()
