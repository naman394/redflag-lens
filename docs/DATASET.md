# RedFlag Lens: Dataset Plan

No public labelled dataset of risk sentences from Indian annual reports turned up in the search. US 10-K datasets exist but are either weakly labelled (GRAB) or don't share a public data link. So the dataset is something you **build**, and it counts as part of your contribution.

The data has five layers. Only layer 1 needs much human effort.

| # | Layer | What it trains or tests | Size | Effort |
|---|---|---|---|---|
| 1 | **Gold sentences** (your hand labels) | Risk classifier, category classifier, specificity classifier; the final test set | ~2,000–2,500 sentences | High (the main work) |
| 2 | **Silver sentences** (LLM-labelled) | Extra training data only, never testing | 10k–30k sentences | Low (scripted) |
| 3 | **Public helper datasets** | Pre-training, NER evaluation, cross-market comparison | Ready-made | Low |
| 4 | **Report-level sets** (checklist, YoY, backtest) | Checklist accuracy, YoY diff, backtest demo | ~40 report-level labels | Medium |
| 5 | **Pair and span sets** | Similarity search, NER/IE evaluation | ~300 pairs, ~400 sentences with spans | Medium |

---

## 1. Raw material: which reports to collect

### Where to get them
- **NSE and BSE company pages** (the "Annual Reports" section under corporate filings). These are free PDFs.
- **Company investor-relations websites**, often with older years.
- **`indianfinance` (PyPI)** has an `annual_report(years)` function built on NSE's unofficial endpoints. It may break, so treat it as a convenience, not a guarantee. Throttle requests and respect the sites' terms.
- MCA21 filings (AOC-4) are another source but need paid access, so skip them for the course.
- Log every PDF in `data/raw/manifest.csv` with columns `company, cin_or_symbol, sector, fy, url, download_date, sha256, pages, text_layer(Y/N), group(train/val/test/backtest)`.

### What mix to pick (~60 reports total)

| Group | Count | Purpose |
|---|---|---|
| **Healthy companies**, spread across 8–10 sectors (manufacturing, IT, pharma, FMCG, infra, NBFC, bank, auto, real estate, power) | ~25 | Realistic, mostly boilerplate risk text; normal negatives |
| **Distressed companies**: the 1–2 reports **before** a known default, insolvency (IBC) filing, SEBI order or auditor resignation | ~15 | Real red flags such as going-concern doubts, CARO defaults and qualified opinions. Without these, the "Specific" class will be tiny. |
| **Prior-year reports** for 10 of the above (two consecutive years) | ~10 | YoY diff testing |
| **Held-out backtest set**: distressed firms plus a matched healthy firm each (same sector, similar size) | ~10 pairs | Final demo; never used in training |

Candidate distressed firms (dates must be checked one by one): IL&FS, DHFL, Café Coffee Day, Jet Airways, SREI, Reliance Capital, Future Retail, Gensol Engineering, plus smaller firms from IBBI's list of admitted insolvency cases and from SEBI enforcement orders.

**Only keep PDFs with a text layer.** Scanned PDFs need OCR, which adds noise. Log them and leave them out (or treat them as a robustness test later).

### Don't label whole reports
A report has 3,000–10,000 sentences, and most are irrelevant. Extract text from the **high-signal sections** and label those:

| Section | Why |
|---|---|
| Independent Auditor's Report (opinion, "Emphasis of Matter", "Material uncertainty related to going concern", Key Audit Matters) | Strongest red flags |
| CARO 2020 annexure | Defaults, unpaid statutory dues, fraud reported, use of short-term funds for long-term purposes |
| Notes on contingent liabilities, borrowings, related-party transactions, going concern | Amounts, counterparties, litigation |
| Board's Report, MD&A, Risk Management | Mostly boilerplate, which is good for training the Generic vs Specific label |
| Corporate Governance report, Secretarial Audit (MR-3) | SEBI/stock-exchange penalties, director resignations, non-compliance |

---

## 2. Layer 1: gold sentence labels (the core dataset)

### Label schema (one row per sentence)

```
sentence_id, company, fy, page, section, sentence,
is_risk          (0/1)
category         Financial | Legal | Regulatory | Operational | Business | None
category_2       optional secondary category
specificity      Generic | Specific | None
annotator, split
```

**Specificity** is the most important label for the product. A *Generic* risk is boilerplate that any company could write: "We face intense competition." A *Specific* risk names a concrete event, amount, counterparty or regulator action: "The Company has defaulted on repayment of ₹212 crore to State Bank of India." Specific risks are the red flags.

### Example rows

| Sentence | is_risk | category | specificity |
|---|---|---|---|
| "The Company is exposed to fluctuations in foreign exchange rates." | 1 | Financial | Generic |
| "The Company has defaulted in repayment of dues to banks amounting to ₹212.4 crore, overdue since 2 March 2019." | 1 | Financial | Specific |
| "SEBI imposed a penalty of ₹10 lakh on the Company for delayed disclosure under Regulation 30." | 1 | Regulatory | Specific |
| "A customer has filed a suit claiming damages of ₹8.5 crore, which the Company is contesting." | 1 | Legal | Specific |
| "These events indicate a material uncertainty that may cast significant doubt on the Company's ability to continue as a going concern." | 1 | Financial | Specific |
| "Production at the Pune plant was suspended for 41 days due to a fire." | 1 | Operational | Specific |
| "The Company operates five manufacturing plants across India." | 0 | None | None |
| "Revenue increased by 12% to ₹1,420 crore." | 0 | None | None |

### Category decision rules (put these in `annotations/guide.md`)
- **Regulatory** means a regulator or government is the actor (SEBI, RBI, GST, tax authorities, environmental bodies): penalties, show-cause notices, licence issues.
- **Legal** means a court or private party is the actor: suits, arbitration, NCLT petitions filed by creditors.
- Where both apply (e.g. a SEBI order under appeal at SAT), use Regulatory as primary and Legal as secondary.
- **Financial** covers liquidity, debt, defaults, going concern, impairments, unpaid dues, rating downgrades and FX/interest-rate exposure.
- **Operational** covers internal things: plants, supply chain, IT and cyber, people, safety, fraud by employees.
- **Business** covers external markets: competition, demand, customer concentration, pricing, technology change.
- Forward-looking hedges ("may", "could") are usually Generic. Past or present facts with amounts or names are usually Specific.

### How many, and how to sample
- **Target: ~2,200 sentences.** Train ~1,500 / Val ~350 / Test ~350, split **by company** (no company appears in two splits).
- **Train/val pool, sampled for balance:** ~40% from auditor/CARO/contingent-liability sections, ~20% chosen by keyword (default, penalty, litigation, going concern, NCLT, show cause, etc.), ~40% random. This stops the "None" class from swamping everything.
- **Test set, natural distribution:** label **every sentence** in the high-signal sections of the test companies. This reflects real use and keeps precision honest.
- Expect about 35–45% risk sentences in the train pool and 10–20% in the test set.

### Labelling workflow
1. **Tool:** Label Studio (free, self-hosted). Use a text-classification template with three choice fields.
2. **LLM pre-labels the train/val pool** using the guide's definitions with structured JSON output. Humans accept or fix each one, which is roughly 3× faster than labelling from scratch.
3. **The test set is labelled by humans without LLM suggestions**, to avoid anchoring bias (MulFSA reported this problem).
4. **Agreement check:** 200 sentences labelled independently by 2 people → Cohen's kappa for each field. Aim for ≥ 0.6. If it's lower, revise the guide and re-label.
5. **Effort estimate:** about 25–35 person-hours in total. Split across 3–4 people, that's 2 weeks.

### Labelling-quality checks before training
- Remove duplicate and near-duplicate sentences across splits (MinHash or cosine > 0.95). Reports reuse the same text year after year.
- Check that each class has at least ~80 examples in train. Operational and Business may be small, so oversample them from the keyword pool.
- Record the counts and kappa in the data card.

---

## 3. Layer 2: silver labels (cheap extra training data)

- Run the LLM labeller on the high-signal sections of **another 50–100 reports** (train companies only) to get 10k–30k labelled sentences.
- Use them to **pre-train** the classifier, then fine-tune it on gold. Or use them for distillation: a small fast model learns to mimic the LLM.
- **Never put silver labels in val or test.**
- This gives you a clean experiment for M3/M4: *gold only* vs *silver only* vs *silver then gold*.

---

## 4. Layer 3: public datasets that help

| Dataset | Use it for | Notes |
|---|---|---|
| **Financial PhraseBank** (Malo et al.) | Warm-up fine-tuning for sentiment and tone in finance; a sanity check that your training code works | ~4,800 news sentences, CC BY-NC-SA 3.0 |
| **FiNER-ORD** | Gold set for **evaluating** ORG/PERSON/LOC NER on financial text before you label your own spans | CC BY-NC 4.0 |
| **FiNER-139** | Optional; shows that BERT handles numbers badly (supports using rules for ₹ amounts) | XBRL numeric tags, US filings |
| **EDGAR-CORPUS** (`eloukas/edgar-corpus` on Hugging Face) | Item 1A risk-factor text: unlabelled domain text, extra silver positives, and a **US → India transfer** experiment | US 10-Ks 1993–2020; check the licence on the dataset card |
| **NAACL-2018 10-K risk sentences** (N18-5017) | Human-labelled risk/non-risk sentences as extra binary data | Check whether the data is downloadable before planning around it |
| **GRAB** (arXiv 2509.21698) | Borrow its taxonomy (21 fine types, 5 macro classes) to justify your categories | The paper says data and code are released, but no repository link was found; treat it as a citation unless you find it |
| **Pre-trained models** (not data): `ProsusAI/finbert`, `yiyanghkust/finbert-pretrain`, spaCy `en_core_web_trf`, `BAAI/bge-base-en-v1.5` | Starting checkpoints | These save you from training from scratch |

Non-commercial licences (CC BY-NC) are fine for a course project. State them in the data card, and note that a commercial product would need its own data.

---

## 5. Layer 4: report-level sets (checklist, YoY, backtest)

### Checklist gold
For **10 test reports**, a human fills in the ~20-item red-flag checklist (Found / Not found + page). This measures checklist precision and recall. It needs about 1 hour per report.

```
report_id, item_id, item_name, present(Y/N), page, evidence_sentence
```

### YoY pairs
For the **10 two-year pairs**, label each risk in year *t* as `New | Removed | Escalated | Unchanged` relative to *t−1*. This evaluates the diff feature.

### Backtest set
```
company, sector, fy_report, event, event_date, event_source_url, matched_control_company
```
- **event** is a default, IBC admission, SEBI order, auditor resignation or rating downgrade to D.
- Use the report filed **1–2 years before** the event.
- Keep these companies out of all training.

---

## 6. Layer 5: pair and span sets

- **Similarity pairs (~300):** pick a risk sentence, retrieve its top-5 candidates from the same report, and label each one related or not related. This gives Precision@5, MRR and ROC-AUC.
- **Spans (~400 risk sentences from the test set):** ORG, PERSON, LOC, MONEY, DATE, EVENT_TRIGGER. Use Label Studio's NER template and export as CoNLL. Evaluate with `seqeval` for entity-level P/R/F1.
- **₹ normaliser test file (~150 strings):** `₹8.5 crore`, `Rs. 85,000 lakh`, `INR 1,23,45,678`, `₹ (12.40) Cr.`, broken-glyph versions → expected value in rupees. These are unit tests rather than ML data, but put them in the repo.

---

## 7. Final folder layout

```
data/
  raw/                 PDFs (git-ignored) + manifest.csv
  interim/             extracted text per page, section tags (JSONL)
  annotations/
    guide.md
    gold_sentences.jsonl     layer 1
    agreement_200.jsonl      double-labelled subset
    silver_sentences.jsonl   layer 2
    spans.conll              layer 5
    sim_pairs.csv            layer 5
    checklist_gold.csv       layer 4
    yoy_gold.csv             layer 4
    backtest.csv             layer 4
  splits/              train/val/test company lists (fixed, committed)
  DATA_CARD.md
```

## 8. Order of work for M2
1. Collect 60 PDFs and fill in the manifest (days 1–3).
2. Extract text and tag sections, then spot-check 5 reports by hand (days 3–5).
3. Write guide v1. Two people label the same 200 sentences → kappa → guide v2 (days 5–7).
4. LLM pre-labels the train pool, and humans correct it. Humans label the test set without suggestions (week 2).
5. Remove duplicates, freeze splits, write the data card and EDA notebook (end of week 2).
6. Silver labels, spans, pairs and checklist gold can follow in M3/M4 as needed.

## Sources
- GRAB: https://arxiv.org/abs/2509.21698
- Taxonomy-aligned 10-K risk extraction (LLM + embeddings + judge; S&P 500, ~10.7k risk factors): https://arxiv.org/abs/2601.15247
- EDGAR-CORPUS: https://huggingface.co/datasets/eloukas/edgar-corpus
- FiNER-ORD: https://arxiv.org/abs/2302.11157
- NAACL-2018 risk sentences: https://aclanthology.org/N18-5017
- LLM annotators / anchoring bias: https://arxiv.org/abs/2403.18152 · https://arxiv.org/abs/2504.02429
- indianfinance (unofficial NSE downloader): https://pypi.org/project/indianfinance/
