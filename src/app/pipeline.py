"""RedFlag Lens analysis pipeline: annual-report PDF -> red-flag findings with page evidence.

Stages (each timed and logged):
  1. validate   reject encrypted / corrupt / scanned / oversized PDFs with a clear message
  2. extract    text blocks + section tags (src/prep/02_extract.py)
  3. sentences  split, clean, drop fragments (03_sentences.py, 03b_flag_fragments.py)
  4. classify   risk / category / specificity (FinBERT if trained, else TF-IDF + LR)
  5. enrich     ₹ amounts and dates per finding
  6. summarise  LLM writes claims that must cite finding ids; a verifier drops any claim whose
                citations don't exist or whose ₹ amounts don't appear in the cited sentences.
                If the LLM fails, a template summary is built from the findings instead.
  7. monitor    one JSON line per report in logs/reports.jsonl
"""
import hashlib, importlib.util, json, os, re, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models"))
LOGS = ROOT / "logs"
MAX_PAGES = 700
HIGH_CONF = 0.5  # default view: findings the model is confident about
LOW_CONF = (0.35, 0.65)


def _mod(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EXTRACT = _mod("src/prep/02_extract.py", "extract")
SENTS = _mod("src/prep/03_sentences.py", "sentences")
FRAG = _mod("src/prep/03b_flag_fragments.py", "fragments")
SKIP = SENTS.SKIP_SECTIONS
from rules import is_nil_statement, is_severe, is_template, norm  # noqa: E402  (src/models on sys.path)

_BP = Path(__file__).resolve().parents[2] / "models/boilerplate.txt"
BOILERPLATE = set(_BP.read_text().split("\n")) if _BP.exists() else set()


def demote_reason(text: str) -> str:
    """Why a confident prediction is moved to the review tier: questionnaire/template rows, or wording shared by
    >= 3 training companies (standard text, not a company-specific event)."""
    if is_template(text):
        return "template"
    if norm(text) in BOILERPLATE:
        return "boilerplate"
    return ""


class ReportError(Exception):
    """A PDF we can't analyse, with a message for the user."""


# ---------------------------------------------------------------- models
class Classifier:
    def __init__(self, prefer: str = "finbert"):
        import joblib
        self.tfidf = joblib.load(ROOT / "models/tfidf_lr.joblib")
        self.name = "tfidf_lr"
        self.threshold = self.tfidf["threshold"]
        # "finbert" (default, fast on a laptop CPU) or "deberta" (DeBERTa-v3-large + aug: more accurate, ~3x slower)
        fb = ROOT / {"finbert": "models/finbert", "deberta": "models/debv3l_aug"}.get(prefer, "models/finbert")
        if prefer in ("finbert", "deberta") and fb.exists():
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(fb)
            self.model = AutoModelForSequenceClassification.from_pretrained(fb, dtype=torch.float32).to(self.dev).eval()
            meta = json.loads((fb / "meta.json").read_text())
            self.labels, self.threshold, self.name = meta["labels"], meta["threshold"], prefer

    def _tf(self, texts):
        from scipy.sparse import hstack
        return hstack([v.transform(texts) for v in self.tfidf["vecs"]]).tocsr()

    def predict(self, inputs: list[str]):
        X = self._tf(inputs)
        spec = self.tfidf["spec"].predict(X)
        if self.name == "tfidf_lr":
            score = self.tfidf["risk"].predict_proba(X)[:, 1]
            cat = self.tfidf["cat"].predict(X)
        else:
            probs = []
            with self.torch.no_grad():
                for i in range(0, len(inputs), 64):
                    enc = self.tok(inputs[i:i + 64], truncation=True, max_length=128, padding=True, return_tensors="pt").to(self.dev)
                    probs.append(self.torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
            P = np.concatenate(probs) if probs else np.zeros((0, len(self.labels)))
            score = 1 - P[:, 0]
            cat = np.array(self.labels[1:])[P[:, 1:].argmax(1)] if len(P) else np.array([])
        return score, cat, spec


class Cascade:
    """Two-step CPU setup (src/eval/cascade.py): FinBERT scores every sentence and gives the category; only sentences
    with FinBERT score >= GATE are re-scored by DeBERTa-v3-large + aug. Test v1.1: F1 0.83 vs 0.77 for FinBERT alone."""
    GATE = 0.05  # = FinBERT's validation-tuned review threshold, not tuned on test

    def __init__(self):
        self.fb, self.db = Classifier("finbert"), Classifier("deberta")
        assert self.fb.name == "finbert" and self.db.name == "deberta", "cascade needs models/finbert and models/debv3l_aug"
        self.name, self.threshold = "cascade", self.fb.threshold

    def predict(self, inputs: list[str]):
        score, cat, spec = self.fb.predict(inputs)
        idx = np.flatnonzero(score >= self.GATE)
        if len(idx):
            score = score.copy()
            score[idx] = self.db.predict([inputs[i] for i in idx])[0]
        return score, cat, spec


_CLF = None


def classifier():
    global _CLF
    if _CLF is None:
        m = os.environ.get("REDFLAG_MODEL", "finbert")
        _CLF = Cascade() if m == "cascade" else Classifier(m)
    return _CLF


# ---------------------------------------------------------------- stages
def validate(pdf_bytes: bytes):
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise ReportError("This file could not be opened as a PDF. It may be corrupt.")
    if doc.needs_pass:
        raise ReportError("This PDF is password-protected. Please upload an unlocked copy.")
    if doc.page_count > MAX_PAGES:
        raise ReportError(f"This PDF has {doc.page_count} pages; the limit is {MAX_PAGES}. Upload the annual report only.")
    sample = [doc[i] for i in range(0, doc.page_count, max(1, doc.page_count // 20))]
    chars = sum(len(p.get_text()) for p in sample) / max(len(sample), 1)
    if chars < 200:
        raise ReportError("This PDF looks scanned (almost no text layer). OCR it first, or upload the text version from BSE/NSE.")
    return doc


def sentences(doc, meta: dict) -> list[dict]:
    blocks = [b for b in EXTRACT.extract_doc(doc, meta) if not b["is_table"] and b["section"] not in SKIP]
    merged, buf = [], None
    for b in blocks:  # same joining rule as 03_sentences.merged_blocks
        if (buf and buf["section"] == b["section"] and b["page"] - buf["page"] <= 1
                and not re.search(r"[.:;?!)\"]$", buf["text"]) and re.match(r"^[a-z(]", b["text"])):
            buf["text"] += " " + b["text"]
            continue
        if buf:
            merged.append(buf)
        buf = dict(b)
    if buf:
        merged.append(buf)
    out, seen = [], set()
    for b in merged:
        for s in SENTS.split(SENTS.fix_text(b["text"])):
            s = re.sub(r"\s+", " ", s).strip()
            key = re.sub(r"\W+", " ", s.lower()).strip()
            if not SENTS.keep(s) or key in seen or FRAG.reason(s):
                continue
            seen.add(key)
            out.append({"id": len(out), "page": b["page"], "section": b["section"], "scope": b["scope"], "text": s})
    return out


MONEY = SENTS.MONEY_RE
DATE = SENTS.DATE_RE


def enrich(f: dict) -> dict:
    f["amounts"] = [m.group(0).strip() for m in MONEY.finditer(f["text"])]
    f["dates"] = [m.group(0).strip() for m in DATE.finditer(f["text"])]
    return f


def RANK(f):
    return (bool(f.get("severe")), f["specificity"] == "Specific", f["score"])


def template_summary(findings: list[dict]) -> list[dict]:
    """Deterministic fallback: one claim per category from its highest-scoring specific finding."""
    claims = []
    severe = [f for f in findings if f.get("severe")]
    if severe:
        top = severe[:3]
        claims.append({"claim": f"{len(severe)} severe warning sign(s) (going concern, default, audit qualification, "
                                f"insolvency, fraud or similar). Most serious: " +
                                "; ".join(f"p.{f['page']}: \"{f['text'][:140]}\"" for f in top),
                       "evidence": [f["id"] for f in top], "verified": True})
    by_cat = {}
    for f in findings:
        by_cat.setdefault(f["category"], []).append(f)
    for cat, fs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        top = sorted(fs, key=RANK, reverse=True)[0]
        claims.append({"claim": f"{len(fs)} {cat.lower()} red flag(s); most notable on p.{top['page']}: \"{top['text'][:180]}\"",
                       "evidence": [top["id"]], "verified": True})
    return claims


def llm_summary(findings: list[dict], company: str):
    """Ask Gemini for 3-6 claims citing finding ids, then verify every claim against its evidence."""
    from typing import List
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class Claim(BaseModel):
        claim: str
        evidence: List[int]

    class Summary(BaseModel):
        claims: List[Claim]

    top = sorted(findings, key=RANK, reverse=True)[:40]
    listing = "\n".join(f"[{f['id']}] (p.{f['page']}, {f['section']}, {f['category']}) {f['text']}" for f in top)
    prompt = (f"These are red-flag sentences found in the annual report of {company}. Write 3-6 short claims for a "
              "credit analyst, most serious first. Every claim must cite the ids of the sentences that support it, "
              "use only facts stated in those sentences, and copy any ₹ amount exactly as written.\n\n" + listing)
    client = genai.Client()
    resp = client.models.generate_content(
        model=os.environ.get("REDFLAG_LLM", "gemini-4-neon"), contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Summary))
    parsed = resp.parsed if resp.parsed is not None else Summary.model_validate_json(resp.text)
    by_id = {f["id"]: f for f in findings}
    out, rejected = [], 0
    for c in parsed.claims:
        ev = [i for i in c.evidence if i in by_id]
        evidence_text = " ".join(by_id[i]["text"] for i in ev)
        nums = re.findall(r"\d[\d,]*\.?\d*", " ".join(m.group(0) for m in MONEY.finditer(c.claim)))
        ok = bool(ev) and len(ev) == len(c.evidence) and all(n in evidence_text for n in nums)
        if ok:
            out.append({"claim": c.claim, "evidence": ev, "verified": True})
        else:
            rejected += 1
    return out, rejected


def analyze(pdf_bytes: bytes, company: str = "Uploaded report", fy: str = "", use_llm: bool = True) -> dict:
    t0, timing = time.perf_counter(), {}

    def lap(name):
        nonlocal t0
        timing[name] = round(time.perf_counter() - t0, 2); t0 = time.perf_counter()

    doc = validate(pdf_bytes); lap("validate")
    sents = sentences(doc, {"company": company, "fy": fy, "group": ""}); lap("extract_sentences")
    clf = classifier()
    score, cat, spec = clf.predict([f"[{s['section']}] {s['text']}" for s in sents]); lap("classify")

    findings = []
    for s, sc, c, sp in zip(sents, score, cat, spec):
        # two tiers: "high" (>= 0.5) shown by default; "review" (>= the val threshold tuned for recall >= 0.85)
        if sc >= min(clf.threshold, HIGH_CONF) and not is_nil_statement(s["text"]):
            why = demote_reason(s["text"])
            tier = "high" if sc >= HIGH_CONF and not why else "review"
            findings.append(enrich({**s, "score": round(float(sc), 3), "category": str(c), "specificity": str(sp),
                                    "tier": tier, "demoted": why, "severe": is_severe(s["text"])}))
    findings.sort(key=RANK, reverse=True)
    high = [f for f in findings if f["tier"] == "high"]
    lap("enrich")

    summary, rejected, llm_error, summary_source = [], 0, None, "none"
    if findings:
        if use_llm and os.environ.get("GEMINI_API_KEY"):
            try:
                summary, rejected = llm_summary(high or findings, company)
                summary_source = "llm_verified"
            except Exception as e:  # LLM down / quota / bad output -> fall back, never fail the report
                llm_error = f"{type(e).__name__}: {str(e)[:200]}"
        if not summary:
            summary, summary_source = template_summary(high or findings), "template"
    lap("summarise")

    cats = Counter(f["category"] for f in high)
    low = sum(LOW_CONF[0] <= s <= LOW_CONF[1] for s in score)
    result = {
        "company": company, "fy": fy, "pages": doc.page_count, "sentences_analysed": len(sents),
        "model": clf.name, "threshold": clf.threshold,
        "high_conf": HIGH_CONF, "n_findings": len(high), "n_review": len(findings) - len(high),
        "n_specific": sum(f["specificity"] == "Specific" for f in high),
        "n_severe": sum(f["severe"] for f in high), "n_demoted": sum(bool(f["demoted"]) for f in findings),
        "by_category": dict(cats), "summary": summary, "summary_source": summary_source,
        "findings": findings, "timing_s": timing, "total_s": round(sum(timing.values()), 2),
        "message": None if high else "No red-flag sentences were found above the threshold. This is not a guarantee: review the auditor's report manually.",
    }
    monitor(pdf_bytes, result, score, low, rejected, llm_error)
    return result


def monitor(pdf_bytes, r, score, low, rejected, llm_error):
    LOGS.mkdir(exist_ok=True)
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "sha1": hashlib.sha1(pdf_bytes).hexdigest()[:12],
           "company": r["company"], "pages": r["pages"], "sentences": r["sentences_analysed"], "model": r["model"],
           "findings": r["n_findings"], "by_category": r["by_category"],
           "finding_rate": round(r["n_findings"] / max(r["sentences_analysed"], 1), 4),
           "mean_score": round(float(np.mean(score)), 4) if len(score) else None,
           "low_conf_share": round(low / max(len(score), 1), 4), "verifier_rejected": rejected,
           "summary_source": r["summary_source"], "llm_error": llm_error, "timing_s": r["timing_s"]}
    with open(LOGS / "reports.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def page_image(pdf_bytes: bytes, page: int, highlight: str, zoom: float = 1.6) -> bytes:
    """PNG of one page with the evidence sentence highlighted."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    p = doc[page - 1]
    words = highlight.split()
    for chunk in [" ".join(words[i:i + 6]) for i in range(0, len(words), 6)]:
        for rect in p.search_for(chunk):
            p.add_highlight_annot(rect)
    return p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


if __name__ == "__main__":
    path = Path(sys.argv[1])
    r = analyze(path.read_bytes(), company=path.stem)
    print(json.dumps({k: v for k, v in r.items() if k != "findings"}, indent=1, ensure_ascii=False))
    for f in r["findings"][:10]:
        print(f"p.{f['page']:<4} {f['category']:<12} {f['specificity']:<9} {f['score']:.2f}  {f['text'][:120]}")
