"""Step 3b: flag table fragments and broken sentences (no rows are removed, so sentence_ids stay stable).

Adds two fields to every row of data/interim/sentences.jsonl:
  is_fragment      True if the text looks like a table row, column header or broken piece of a sentence
  fragment_reason  which rule fired (empty if not a fragment)

Downstream steps (labelling samples, training, inference) skip rows with is_fragment = True.

  python src/prep/03b_flag_fragments.py            # flag the corpus in place
  python src/prep/03b_flag_fragments.py --eval     # check the rules against the pilot labels only
"""
import argparse, gzip, json, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENT = ROOT / "data/interim/sentences.jsonl"
PILOT = ROOT / "data/annotations/pilot_300_labelled.jsonl"

NUM_RE = re.compile(r"^[(\-–]?[\d,]*\d(\.\d+)?%?[)*]?$|^\d{4}-\d{2,4}$|^[-–]$")
# Words that make a clause: auxiliaries, modals, common reporting verbs, and -ed/-ing forms.
VERB_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|will|shall|may|might|can|could|would|should|must|"
    r"does|do|did|ensure|ensures|include|includes|remain|remains|require|requires|relate|relates|"
    r"comprise|comprises|consist|consists|represent|represents|refer|refers|exist|exists|"
    r"\w{3,}ed|\w{3,}ing)\b", re.I)


STOP = set("""a an the of to in on at by for with from and or but as is are was were be been has have had
will shall may can could would should not no its it this that these those which who whose any such than into
during under upon over against between our their his her we they there per""".split())


def reason(text: str) -> str:
    toks = text.split()
    nums = sum(bool(NUM_RE.match(t)) for t in toks)
    decimals = len(re.findall(r"\b\d[\d,]*\.\d+\b", text))
    has_verb = bool(VERB_RE.search(text))
    words = [t for t in toks if t[:1].isalpha()]
    title = sum(t[0].isupper() for t in words) / max(len(words), 1)
    stop = sum(t.lower().strip(",;:()") in STOP for t in toks) / len(toks)
    ends_number = bool(re.search(r"\d[\d,]*(\.\d+)?\)?\s*$", text))

    if re.match(r"^[,;:.)]", text):
        return "starts_with_punct"
    if nums / len(toks) >= 0.25:
        return "numeric_row"
    ends_sentence = bool(re.search(r"[.?!][)\"']?$", text))
    if decimals >= 2 and stop < 0.2 and not ends_sentence:
        return "numeric_row"
    if ends_number and (decimals >= 1 or nums >= 2):  # a row of figures: text label then values
        return "numeric_row"
    if title >= 0.7 and stop < 0.25 and not ends_sentence:
        return "header_row"
    if title >= 0.7 and stop < 0.15 and not has_verb:  # e.g. "13 CIL NTPC Urja Private Ltd."
        return "header_row"
    if not has_verb and not re.search(r"[.?!:]$", text) and stop < 0.2:
        return "no_clause"
    return ""


def evaluate() -> None:
    rows = [json.loads(l) for l in open(PILOT)]
    tp = fp = fn = risk_hit = 0
    for r in rows:
        bad = "BAD_EXTRACTION" in (r.get("notes") or "")
        why = reason(r["text"])
        tp += bad and bool(why); fn += bad and not why; fp += (not bad) and bool(why)
        if why and int(r["is_risk"]) == 1:
            risk_hit += 1
            print(f"  !! RISK sentence flagged ({why}): {r['text'][:110]}")
        if why and not bad:
            print(f"  fp ({why}): {r['text'][:110]}")
        if bad and not why:
            print(f"  missed: {r['text'][:110]}")
    p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    print(f"\npilot n={len(rows)}  BAD_EXTRACTION={tp + fn}  flagged={tp + fp}")
    print(f"precision={p:.2f}  recall={rc:.2f}  risk sentences flagged={risk_hit}")


def flag_corpus() -> None:
    tmp = SENT.with_suffix(".tmp")
    n = flagged = 0
    by_reason = {}
    with open(SENT) as f, open(tmp, "w") as out:
        for line in f:
            r = json.loads(line)
            why = reason(r["text"])
            r["is_fragment"], r["fragment_reason"] = bool(why), why
            n += 1; flagged += bool(why)
            by_reason[why] = by_reason.get(why, 0) + 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(SENT)
    with open(SENT, "rb") as f, gzip.open(str(SENT) + ".gz", "wb") as g:
        shutil.copyfileobj(f, g)
    print(f"sentences: {n}  flagged as fragments: {flagged} ({flagged / n:.1%})")
    for k, v in sorted(by_reason.items(), key=lambda x: -x[1]):
        if k:
            print(f"  {k:18} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate() if a.eval else flag_corpus()
