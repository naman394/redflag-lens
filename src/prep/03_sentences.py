"""Step 3: split text blocks into sentences, filter noise, add simple features.

Output: data/interim/sentences.jsonl
  {sentence_id, company, fy, group, page, section, scope, text, n_words, has_money, has_date, kw_hit}
"""
import json, re
from pathlib import Path

import pysbd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data/interim/blocks.jsonl"
OUT = ROOT / "data/interim/sentences.jsonl"

SKIP_SECTIONS = {"TOC", "FRONT", "NOTICE", "FIN_STATEMENTS"}  # cover pages, AGM notice, number tables

# Abbreviations common in Indian reports that must not end a sentence.
ABBREV = ["Rs.", "No.", "Nos.", "Ltd.", "Pvt.", "Co.", "Corpn.", "viz.", "i.e.", "e.g.", "etc.", "Mr.", "Ms.", "Mrs.", "Dr.", "Sr.", "Jr.", "Govt.", "Dept.", "approx.", "Reg.", "Regn.", "Sec.", "Cl.", "Para.", "Ref.", "Vol.", "Bros.", "Inc.", "St."]
PROTECT = {a: a.replace(".", "<DOT>") for a in ABBREV}

MONEY_RE = re.compile(r"(₹|\bRs\.?|\bINR)\s*\(?\s*[\d,]+(\.\d+)?\s*\)?\s*(crores?|cr\.?|lakhs?|lacs?|million|mn|billion|bn|thousand)?|\b[\d,]+(\.\d+)?\s*(crores?|lakhs?|lacs?)\b", re.I)
DATE_RE = re.compile(r"\b(\d{1,2}(st|nd|rd|th)?\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+\d{4}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\bFY\s?\d{2,4}(-\d{2,4})?\b|\b(19|20)\d{2}-\d{2}\b", re.I)
RISK_KW_RE = re.compile(
    r"going concern|default|overdue|penalt|show cause|litigation|lawsuit|suit filed|arbitration|claim|"
    r"NCLT|NCLAT|insolvency|IBC|winding up|fraud|wilful|invok|downgrad|impair|write.?off|provision for|"
    r"qualified opinion|emphasis of matter|material uncertainty|adverse|non.?compliance|contraven|"
    r"SEBI|RBI|enforcement directorate|investigation|search and seizure|disputed|demand notice|"
    r"breach|covenant|restructur|moratorium|NPA|non.?performing|liquidity|negative net worth|erosion|"
    r"pledge|resign|unsecured|guarantee", re.I)

seg = pysbd.Segmenter(language="en", clean=False)


def fix_text(text: str) -> str:
    """Remove control characters and repair broken ligatures ("fi nancial" -> "financial")."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f\ufffd]", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(fi|fl|ff|ffi) (?=[a-z])", r"\1", text)   # identifi ed -> identified
    text = re.sub(r"\b(fi|fl|Fi|Fl) (?=[a-z]{2,})", r"\1", text)          # fi led -> filed
    return text


def split(text: str) -> list[str]:
    for a, p in PROTECT.items():
        text = text.replace(a, p)
    out = []
    for s in seg.segment(text):
        s = s.replace("<DOT>", ".").strip()
        # break enumerated clause lists like "(a) ... ; (b) ..."
        out.extend(x.strip() for x in re.split(r";\s+(?=\(?[a-z]{1,3}\)|\(?[ivx]{1,4}\))", s) if x.strip())
    return out


def keep(s: str) -> bool:
    words = s.split()
    if not 6 <= len(words) <= 120:
        return False
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    if letters < 0.55 * len(s) or digits > 0.2 * len(s):
        return False
    if re.match(r"^(place|date|for and on behalf|sd/-|din\b|membership no|firm registration|partner\b|chairman\b)", s, re.I):
        return False
    return True


def main() -> None:
    n_in = n_out = n_dup = 0
    seen = set()
    counter = {}
    def merged_blocks(f):
        """Join a block that ends mid-sentence with the next block if that one continues in lowercase."""
        buf = None
        for line in f:
            b = json.loads(line)
            if b["is_table"] or b["section"] in SKIP_SECTIONS:
                continue
            if (buf and buf["company"] == b["company"] and buf["section"] == b["section"]
                    and b["page"] - buf["page"] <= 1 and not re.search(r"[.:;?!)\"]$", buf["text"])
                    and re.match(r"^[a-z(]", b["text"])):
                buf["text"] += " " + b["text"]
                continue
            if buf:
                yield buf
            buf = b
        if buf:
            yield buf

    with open(IN) as f, open(OUT, "w") as out:
        for b in merged_blocks(f):
            n_in += 1
            for s in split(fix_text(b["text"])):
                s = re.sub(r"\s+", " ", s).strip()
                if not keep(s):
                    continue
                key = (b["company"], b["fy"], re.sub(r"\W+", " ", s.lower()).strip())
                if key in seen:  # exact repeat inside the same report (e.g. standalone vs consolidated notes)
                    n_dup += 1
                    continue
                seen.add(key)
                cid = f"{b['company']}_{b['fy']}"
                counter[cid] = counter.get(cid, 0) + 1
                row = {
                    "sentence_id": f"{cid}_{counter[cid]:05d}",
                    "company": b["company"], "fy": b["fy"], "group": b["group"], "page": b["page"],
                    "section": b["section"], "scope": b["scope"], "text": s, "n_words": len(s.split()),
                    "has_money": bool(MONEY_RE.search(s)), "has_date": bool(DATE_RE.search(s)),
                    "kw_hit": bool(RISK_KW_RE.search(s)),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"text blocks (after joining): {n_in}  sentences kept: {n_out}  exact repeats removed: {n_dup}")
    for k, v in sorted(counter.items()):
        print(f"  {k:24} {v}")


if __name__ == "__main__":
    main()
