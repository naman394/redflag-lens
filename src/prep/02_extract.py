"""Step 2: extract clean text blocks from each PDF and tag the report section they belong to.

Output: data/interim/blocks.jsonl, one row per text block:
  {company, fy, group, page, block_no, section, scope, text, is_table}

Cleaning:
  - rupee symbol: spans in a "Rupee" font (e.g. ITFRupee, Rupee Foradian) become "₹";
    a stray backtick before a number becomes "₹"
  - running headers/footers (top/bottom margin, or lines repeated on many pages) are removed
  - hyphenated line breaks are joined, whitespace and unicode normalised
  - number-heavy blocks are flagged as tables (kept out of the sentence data)
Sections are found from headings (large font) and running headers on each page;
a page keeps the previous page's section until a new heading appears.
"""
import collections, csv, json, re, unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
INBOX = ROOT / "data/raw/inbox"
OUT = ROOT / "data/interim/blocks.jsonl"
STATS = ROOT / "data/interim/section_coverage.csv"

# Order matters: more specific patterns first.
SECTION_PATTERNS = [
    ("SECRETARIAL_AUDIT", r"secretarial\s+audit\s+report|form\s+no\.?\s*mr\s*-?\s*3"),
    ("CARO", r"companies\s*\(auditor.{0,3}s\s+report\)\s*order|annexure\s*.{0,4}[ab].{0,4}\s*to\s+the\s+independent\s+auditor"),
    ("IFC_REPORT", r"report\s+on\s+the\s+internal\s+financial\s+controls"),
    ("AUDITOR_REPORT", r"independent\s+auditor.{0,3}s?\s+report|auditors.{0,2}\s+report\s+to\s+the\s+members|report\s+on\s+the\s+audit\s+of\s+the\s+(standalone\s+|consolidated\s+)?(ind\s+as\s+)?financial\s+statements"),
    ("NOTES", r"notes\s+(to|forming\s+part\s+of|on)\s+(the\s+)?(standalone\s+|consolidated\s+)?(financial\s+statements|accounts)|schedules?\s+(to|forming\s+part\s+of)\s+(the\s+)?(standalone\s+|consolidated\s+)?(financial\s+statements|balance\s+sheet|accounts)|significant\s+accounting\s+policies"),
    ("FIN_STATEMENTS", r"balance\s+sheet\s+as\s+at|statement\s+of\s+profit\s+and\s+loss|cash\s+flow\s+statement|statement\s+of\s+cash\s+flows|statement\s+of\s+changes\s+in\s+equity"),
    ("MDA", r"management.{0,3}s?\s+discussion\s+(and|&)\s+analysis"),
    ("CORP_GOV", r"report\s+on\s+corporate\s+governance|corporate\s+governance\s+report"),
    ("BRR", r"business\s+responsibility(\s+and\s+sustainability)?\s+report"),
    ("BOARDS_REPORT", r"board.{0,3}s\s+report|directors.{0,3}\s+report"),
    ("NOTICE", r"notice\s+is\s+hereby\s+given|notice\s+of\s+(the\s+)?\w+\s+annual\s+general\s+meeting"),
    # any other numbered annexure (CSR report, AOC-2, ...) belongs to the Board's report and ends the previous annexure
    ("BOARDS_REPORT", r"annexure\s*[-:]?\s*['\"]?([a-h]|[ivx]{1,4}|\d{1,2})['\"]?\b"),
]
# Body text that marks the start of the CARO annexure even when its heading is generic
CARO_BODY_RE = re.compile(r"proper\s+records\s+showing\s+full\s+particulars|companies\s*\(auditor.{0,3}s\s+report\)\s*order,?\s*20(16|20)", re.I)
IFC_BODY_RE = re.compile(r"report\s+on\s+the\s+internal\s+financial\s+controls?\s+(over|with\s+reference)", re.I)
SECTION_RES = [(name, re.compile(p, re.I)) for name, p in SECTION_PATTERNS]
CONTINGENT_RE = re.compile(r"contingent\s+liabilit", re.I)
NOTE_HEADING_RE = re.compile(r"^\s*\d{1,2}(\.\d{1,2})?\s*[\.\)]?\s+[A-Z][A-Za-z ,&'/()-]{3,60}$")
RUPEE_FONT_RE = re.compile(r"rupee", re.I)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                      # soft hyphen
    text = re.sub(r"`\s?(?=[\d(])", "₹ ", text)            # stray backtick before a number
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # hyphenated line break
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)              # join lines
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"₹\s*₹", "₹", text)
    return text


def block_text(block: dict) -> tuple[str, float, bool]:
    """Text of a block with rupee-font spans fixed, its largest font size, and whether it is all bold."""
    lines, max_size, all_bold = [], 0.0, True
    for line in block.get("lines", []):
        parts = []
        for span in line["spans"]:
            t = span["text"]
            if RUPEE_FONT_RE.search(span.get("font", "")) and t.strip():
                t = "₹ "
            parts.append(t)
            max_size = max(max_size, span["size"])
            if t.strip() and not (span["flags"] & 16 or "bold" in span.get("font", "").lower()):
                all_bold = False
        lines.append("".join(parts))
    return "\n".join(lines), max_size, all_bold


def is_table_like(text: str) -> bool:
    if not text:
        return True
    digits = sum(c.isdigit() for c in text)
    letters = sum(c.isalpha() for c in text)
    words = text.split()
    return digits > 0.25 * max(len(text), 1) or letters < 0.5 * max(len(text), 1) or len(words) < 4


def main() -> None:
    manifest = {r["file"]: r for r in csv.DictReader(open(ROOT / "data/raw/manifest.csv"))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    coverage = []
    n_blocks = 0
    with open(OUT, "w") as out:
        for fname, meta in sorted(manifest.items()):
            if meta["text_layer"] != "Y":
                continue
            doc = pymupdf.open(INBOX / fname)

            # pass 1: collect pages and find lines repeated on many pages (running headers/footers)
            pages, line_counts = [], collections.Counter()
            for page in doc:
                d = page.get_text("dict")
                h = page.rect.height
                blocks = []
                for b in d["blocks"]:
                    if b.get("type") != 0:
                        continue
                    raw, size, bold = block_text(b)
                    y0, y1 = b["bbox"][1], b["bbox"][3]
                    blocks.append({"raw": raw, "size": size, "bold": bold, "margin": y1 < 0.07 * h or y0 > 0.93 * h})
                pages.append(blocks)
                for key in {normalise(bl["raw"]).lower() for bl in blocks if len(bl["raw"]) < 120}:
                    line_counts[key] += 1
            repeated = {k for k, c in line_counts.items() if c >= max(5, 0.15 * len(pages))}

            # pass 2: section tagging + cleaning
            section, scope, seen = "FRONT", "standalone", collections.Counter()
            for pno, blocks in enumerate(pages, start=1):
                sizes = sorted(bl["size"] for bl in blocks if bl["raw"].strip())
                body = sizes[len(sizes) // 2] if sizes else 10
                titles = [normalise(bl["raw"]) for bl in blocks
                          if bl["margin"] or bl["size"] >= 1.25 * body or (bl["bold"] and len(bl["raw"]) < 150)
                          or normalise(bl["raw"]).lower() in repeated]
                titles = [t for t in titles if len(t) < 200]
                title_text = " | ".join(titles)

                # a heading must match near the start of a title block, not inside a sentence
                hits = []
                for t in titles:
                    for name, rx in SECTION_RES:
                        m = rx.search(t)
                        if m and m.start() <= 15:
                            hits.append(name)
                            break
                if not hits and section in {"AUDITOR_REPORT", "CARO", "IFC_REPORT"}:
                    body_text = " ".join(normalise(bl["raw"]) for bl in blocks)
                    if section != "IFC_REPORT" and IFC_BODY_RE.search(body_text[:1500]):
                        hits = ["IFC_REPORT"]
                    elif section != "CARO" and CARO_BODY_RE.search(body_text):
                        hits = ["CARO"]
                if len(set(hits)) >= 4:                       # table-of-contents page
                    page_section = "TOC"
                else:
                    if hits:
                        # most specific section wins (earliest in SECTION_PATTERNS)
                        order = [n for n, _ in SECTION_PATTERNS]
                        section = min(hits, key=order.index)
                    page_section = section
                if re.search(r"consolidated", title_text, re.I) and page_section in {"AUDITOR_REPORT", "CARO", "IFC_REPORT", "NOTES", "FIN_STATEMENTS"}:
                    scope = "consolidated"
                elif hits and page_section in {"BOARDS_REPORT", "MDA", "CORP_GOV", "NOTICE", "BRR"}:
                    scope = "standalone"
                elif hits and page_section in {"AUDITOR_REPORT", "NOTES", "FIN_STATEMENTS"} and not re.search(r"consolidated", title_text, re.I):
                    scope = "standalone"
                seen[page_section] += 1

                in_contingent = False
                for bno, bl in enumerate(blocks):
                    text = normalise(bl["raw"])
                    if not text or bl["margin"] or text.lower() in repeated:
                        continue
                    sub = page_section
                    if page_section == "NOTES":
                        if CONTINGENT_RE.search(text[:120]):
                            in_contingent = True
                        elif NOTE_HEADING_RE.match(text) and not CONTINGENT_RE.search(text):
                            in_contingent = False
                        if in_contingent:
                            sub = "NOTES_CONTINGENT"
                    row = {
                        "company": meta["company"], "fy": meta["fy"], "group": meta["group"],
                        "page": pno, "block_no": bno, "section": sub, "scope": scope,
                        "text": text, "is_table": is_table_like(text),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_blocks += 1
            coverage.append({"file": fname, "pages": len(pages), **{k: seen.get(k, 0) for k in dict.fromkeys(n for n, _ in SECTION_PATTERNS)}, "TOC": seen.get("TOC", 0), "FRONT": seen.get("FRONT", 0)})
            print(f"{fname:28} pages={len(pages):4} " + " ".join(f"{k[:6]}={seen.get(k,0)}" for k in dict.fromkeys(n for n, _ in SECTION_PATTERNS)))

    with open(STATS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage[0]))
        w.writeheader(); w.writerows(coverage)
    print(f"\n{n_blocks} blocks -> {OUT}")


if __name__ == "__main__":
    main()
