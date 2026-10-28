# Dataset Preparation Plan (who does what, step by step)

Goal: by the end of M2, produce the gold dataset described in [DATASET.md](DATASET.md): ~2,200 labelled sentences from ~60 Indian annual reports, split by company, with a data card and EDA.

**Split of work:** Claude (me) writes and runs all the code: extraction, cleaning, sampling, LLM pre-labelling, quality checks, the data card and EDA. **Your team** does the parts that need human judgement or a browser: approving the company list, downloading PDFs the exchanges won't serve to scripts, correcting labels, and hand-labelling the test set.

---

## Environment facts (checked on this Mac, 2026-10-28)

| Fact | Consequence |
|---|---|
| Python 3.14 is the default; 3.13 and `uv` are installed | Project venv on **Python 3.13 via uv** (some NLP libraries lag on 3.14) |
| **Only ~13 GB free disk**, 8 GB RAM, Apple Silicon | Dataset prep is light (PDFs ≈ 1–2 GB). Don't install PyTorch locally now; **train models on Colab**. PDFs stay git-ignored. |
| NSE/BSE annual-report endpoints refuse automated requests | Get PDFs **manually in a browser** or from **company investor-relations pages / archive.org** with direct PDF links. No bot-check workarounds. |
| No LLM API key in the environment | Needed before step 6 (pre-labelling). Decision below. |

Libraries for prep only: `pymupdf`, `pysbd` (sentence splitting), `spacy` + `en_core_web_sm`, `pandas`, `scikit-learn`, `datasketch` (near-duplicates), `rapidfuzz`, `matplotlib`, `jupyter`, LLM SDK, `label-studio` (run with `uvx`).

---

## Phase 0: Setup (Day 1) [Claude]
- Create the repo structure (`data/`, `src/prep/`, `notebooks/`), `pyproject.toml`, `.gitignore` (PDFs, `.env`), `git init`.
- Write **annotation guide v1** (`data/annotations/guide.md`): definitions, decision rules and 3–4 examples for each category and for Generic/Specific, based on DATASET.md §2.
- **Gate:** your team reads the guide and agrees on it.

## Phase 1: Choose companies and collect PDFs (Days 1–3)
1. **[Claude]** Draft `companies.csv` (~60 rows): company, NSE symbol, sector, fiscal year needed, group (train/val/test/backtest), reason, and for distressed firms the event, event date and source link.
   - Healthy (~25), spread over sectors, for example: TCS, Infosys, HUL, ITC, Asian Paints, Sun Pharma, Cipla, Dr Reddy's, L&T, UltraTech, Tata Steel, JSW Steel, Maruti, M&M, Bajaj Finance, HDFC Bank, NTPC, Tata Power, DLF, Godrej Properties, Titan, Pidilite, Biocon, Wipro, Adani Ports.
   - Distressed (~15) **candidates to verify**: IL&FS group listed companies (IL&FS Transportation, IL&FS Engineering), DHFL, Jet Airways, Café Coffee Day Enterprises, SREI Infrastructure Finance, Reliance Capital, Reliance Communications, Future Retail, Jaiprakash Associates, Videocon Industries, Bhushan Steel, Yes Bank (crisis year), Gensol Engineering. Use the report from **1–2 years before** the event.
   - Backtest pairs (~10): a distressed firm plus a healthy firm of the same sector and size, kept out of training.
2. **[Your team]** Approve the list.
3. **[Team + Claude]** Collect PDFs into `data/raw/inbox/`:
   - Claude fetches what's available as direct PDF links on company websites, or archive.org copies for delisted firms.
   - The team downloads the rest from NSE/BSE company pages in a browser. Expect ~1–2 hours for ~30 PDFs.
   - Name files `SYMBOL_FY.pdf` (e.g. `DHFL_FY2019.pdf`).
4. **[Claude]** `01_ingest.py` builds `manifest.csv`: sha256, page count, file size, **text-layer check** (characters per page), source URL and download date. PDFs that are scanned images or password-protected get flagged and are replaced or dropped.
- **Gate:** ≥ 55 usable PDFs with text layers; every distressed firm has a verified event and source.

## Phase 2: Extract and clean text (Days 3–4) [Claude]
`02_extract.py` (PyMuPDF), one JSONL row per text block with its page number:
- Remove repeated **headers and footers** (lines repeated on more than 30% of pages), page numbers and table-of-contents pages.
- Fix the **₹ glyph** (`` `, `₹`, `■`, `Rs.`, `INR`) → normalised `₹`; join hyphenated line breaks; normalise Unicode and whitespace.
- Flag **table blocks** (high digit/whitespace ratio, short cells) and keep them out of sentence data (tables are a stretch goal).
- **Gate:** I print a before/after sample from 5 reports for you to check.

## Phase 3: Section tagging (Day 4) [Claude]
`03_sections.py` uses heading regexes plus page ranges to tag each block's section:
`AUDITOR_REPORT`, `KAM`, `CARO`, `IFC_REPORT`, `NOTES_CONTINGENT`, `NOTES_BORROWINGS`, `NOTES_RPT`, `NOTES_OTHER`, `BOARDS_REPORT`, `MDA`, `RISK_MGMT`, `CORP_GOV`, `SECRETARIAL_AUDIT`, `OTHER`.
- Report standalone and consolidated statements separately (they repeat each other; keep standalone first).
- **Gate:** coverage table showing, per report, which sections were found. I fix the regexes until the auditor's report, CARO and contingent liabilities are found in ≥ 90% of reports. You spot-check 5 reports against the PDF.

## Phase 4: Sentence segmentation and filtering (Day 5) [Claude]
`04_sentences.py`:
- `pysbd` with a custom rule list so it doesn't split on `Rs.`, `No.`, `Ltd.`, `Pvt.`, `Co.`, `viz.`, `i.e.`, `(i)` or `₹ 8.5`.
- Keep sentences of 6–120 tokens. Drop number-heavy table rows and boilerplate lines (e.g. "Place: Mumbai Date: …").
- Each sentence gets an ID and source info: `sentence_id, company, fy, page, section, char_start, char_end, text`.
- **Gate:** manual check of 100 random sentences for bad splits (target < 5%).

## Phase 5: Splits and sampling (Day 5) [Claude]
`05_sample.py`:
- **Fix the company-wise split first** (train ~70% / val ~15% / test ~15% of companies, stratified by sector and healthy/distressed). Backtest companies are excluded from everything. Commit `splits/*.txt`.
- **Train/val pool (~1,850):** 40% high-signal sections, 20% keyword-seeded (default, penalty, going concern, NCLT, show cause, litigation, wilful, fraud, invoked, …), 40% random.
- **Test pool (~350):** *every* sentence in AUDITOR_REPORT, CARO and NOTES_CONTINGENT, plus a random sample from MDA and CORP_GOV, for the test companies. Natural distribution.
- **Agreement set:** 200 sentences from the train pool for double labelling.
- Remove near-duplicates **before** sampling, both within each split and across splits (MinHash, Jaccard > 0.9).

## Phase 6: LLM pre-labelling (Day 6) [Claude]
`06_prelabel.py`:
- Uses the guide's definitions as the prompt, batches of 20 sentences, temperature 0, structured JSON output (`is_risk, category, category_2, specificity, rationale`).
- Runs on the **train/val pool only**. Test and agreement sets get no suggestions.
- Saves the model name, prompt version and raw outputs (needed for the data card and for reporting "LLM vs human agreement" as a finding).
- Cost: roughly 2k sentences, a few hundred thousand tokens. Small on any model (exact price depends on the model chosen).

## Phase 7: Human labelling (Week 2) [Your team, Claude sets it up]
- **[Claude]** Label Studio project config (three choice fields + an optional comment), task files with **pre-labels** (train/val) and **without** (test, agreement), and a run command (`uvx label-studio`). If Label Studio is too heavy for your disk, I'll build a small Streamlit labelling app instead.
- **[Team]**:
  1. **Agreement round first:** two people independently label the 200-sentence set (~2 hrs each).
  2. **[Claude]** computes Cohen's kappa per field and lists the disagreements. The team discusses them → **guide v2** (and re-labels if kappa < 0.6).
  3. Correct pre-labels on the train/val pool (~1,850; fast, ~10–15 hrs total split across the team).
  4. **Hand-label the test set** (~350, no suggestions; ~4–5 hrs). Ideally two people do it and resolve disagreements.

## Phase 8: Merge and quality checks (end of Week 2) [Claude]
`08_merge_qc.py`:
- Merge exports → `gold_sentences.jsonl`. Check the schema, empty fields and invalid combinations (e.g. `is_risk=0` with a category).
- Class counts per split; flag categories with < 80 training examples and pull extra keyword-seeded sentences for them if needed.
- LLM-vs-human agreement on train/val, per class (a nice result for the report: "how good is LLM pre-labelling?").
- Final cross-split duplicate check; **freeze** the data with a version tag (`data-v1.0`) and checksums.

## Phase 9: Data card and EDA (end of Week 2) [Claude]
- `DATA_CARD.md` (Datasheets for Datasets format): sources, companies, years, sectors, collection dates, guide, annotators, kappa, class balance, known biases (large-cap heavy, English only, text-layer PDFs only), licences, privacy note (public filings; director names appear; the tool profiles companies, not people).
- `notebooks/eda.ipynb`: reports/pages/sentences per company, sentence lengths, sections, label distributions per split, % of sentences with ₹, top terms per category (TF-IDF), sentence reuse across years, and chi-square of category vs section. These cover the M2 "exploratory" and "statistical analysis" items.

## Phase 10: Later extras (M3–M4, as needed) [Claude + team]
- Silver labels: LLM labels on 50–100 more train-company reports (Claude, scripted).
- Spans for NER (~400 test sentences), similarity pairs (~300), checklist gold (10 reports), YoY labels (10 pairs), ₹ normaliser test strings (~150). Claude prepares tasks and pre-fills; the team checks.

---

## Timeline

| When | Claude | Team |
|---|---|---|
| Day 1 | Setup, guide v1, company list draft | Approve guide and list |
| Days 1–3 | Fetch IR/archive PDFs, ingest, manifest | Download remaining PDFs in a browser |
| Days 3–5 | Extraction, sections, sentences, splits, sampling | Spot-check 5 reports and 100 sentences |
| Day 6 | Pre-labelling, Label Studio setup | Agreement round (200 sentences) |
| Days 7–12 | Kappa, guide v2, live progress stats | Correct train/val; hand-label test |
| Days 13–14 | Merge, QC, freeze `data-v1.0`, data card, EDA | Review data card |

## Decisions needed before starting
1. **LLM for pre-labelling:** Claude API, Gemini API, or a local model (slow on 8 GB RAM). An API key must go in `.env`.
2. **Team size,** to divide labelling hours.
3. **Label Studio or a small custom labelling app** (disk space is tight).
4. **Disk:** free a few GB if possible. Prep needs ~2–3 GB; later local model work needs more.
