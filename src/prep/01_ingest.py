"""Step 1: build data/raw/manifest.csv with basic facts about every PDF in data/raw/inbox."""
import csv, hashlib, sys
from datetime import date
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
INBOX = ROOT / "data/raw/inbox"
CANDIDATES = ROOT / "data/raw/candidates.csv"
OUT = ROOT / "data/raw/manifest.csv"

# Companies with a known distress event (default / insolvency / regulator action).
# Event details must be checked against the source before citing them.
DISTRESSED = {
    "JETAIRWAYS": ("Airlines", "Grounded Apr 2019; insolvency (IBC) admitted Jun 2019"),
    "DHFL": ("Housing finance", "Debt defaults from Jun 2019; RBI-initiated insolvency Dec 2019"),
    "RELCAPITAL": ("Diversified finance", "Debt defaults 2020-21; RBI superseded board Nov 2021"),
    "RCOM": ("Telecom", "Insolvency (IBC) filed 2019"),
    "COFFEEDAY": ("Food & beverage", "Founder death Jul 2019; debt defaults; SEBI action"),
    "ILFSTRANS": ("Infrastructure", "IL&FS group defaults Sep 2018"),
    "JPASSOCIAT": ("Infrastructure / cement", "Loan defaults; insolvency petitions"),
    "YESBANK": ("Bank", "RBI moratorium and reconstruction Mar 2020"),
    "GENSOL": ("Renewable energy / EV leasing", "SEBI interim order Apr 2025 alleging fund diversion"),
}
SECTOR = {
    "ASIANPAINT": "Paints", "TCS": "IT services", "INFY": "IT services", "HINDUNILVR": "FMCG",
    "SUNPHARMA": "Pharma", "LT": "Engineering & construction", "MARUTI": "Automobiles",
    "TITAN": "Consumer durables", "PIDILITIND": "Chemicals", "DRREDDY": "Pharma", "NTPC": "Power",
    "ULTRACEMCO": "Cement", "BAJFINANCE": "NBFC", "TATASTEEL": "Steel", "HAVELLS": "Electricals",
    "SRF": "Chemicals",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    urls = {}
    if CANDIDATES.exists():
        for r in csv.DictReader(open(CANDIDATES)):
            urls[r["file"]] = f"https://www.bseindia.com/bseplus/AnnualReport/{r['scrip']}/{r['scrip']}{r['mmyy']}.pdf"

    rows = []
    for pdf in sorted(INBOX.glob("*.pdf")):
        company, fy = pdf.stem.rsplit("_", 1)
        try:
            doc = pymupdf.open(pdf)
        except Exception as e:  # corrupt or not a PDF
            print(f"SKIP {pdf.name}: {e}", file=sys.stderr)
            continue
        chars = [len(p.get_text()) for p in doc]
        empty = sum(c < 50 for c in chars)
        group = "distressed" if company in DISTRESSED else "healthy"
        sector, event = DISTRESSED.get(company, (SECTOR.get(company, "?"), ""))
        rows.append({
            "file": pdf.name, "company": company, "fy": fy, "group": group, "sector": sector,
            "event": event, "pages": doc.page_count, "size_mb": round(pdf.stat().st_size / 1e6, 1),
            "avg_chars_per_page": sum(chars) // max(doc.page_count, 1),
            "pct_pages_no_text": round(100 * empty / max(doc.page_count, 1), 1),
            "text_layer": "Y" if empty / max(doc.page_count, 1) < 0.2 else "N",
            "encrypted": doc.is_encrypted,
            "source_url": urls.get(pdf.stem, "https://www.bseindia.com/xml-data/corpfiling/AttachHis/cf949692-2280-4ee7-ba6f-30d1961f1ad5.pdf" if company == "GENSOL" else ""),
            "download_date": date.today().isoformat(), "sha256": sha256(pdf),
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} PDFs -> {OUT}")
    for r in rows:
        print(f"{r['file']:28} {r['group']:10} pages={r['pages']:4} chars/pg={r['avg_chars_per_page']:5} no_text={r['pct_pages_no_text']:5}% text={r['text_layer']}")


if __name__ == "__main__":
    main()
