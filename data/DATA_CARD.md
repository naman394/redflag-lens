# Data Card: RedFlag Lens Indian Annual-Report Risk Sentences (v0.1, pilot)

*Format follows "Datasheets for Datasets" (Gebru et al.) and Hugging Face dataset cards. Status as of 2026-10-28: corpus built and cleaned; 300-sentence pilot labelled; full gold labelling (~2,200 sentences) scheduled for M3 weeks 1–2.*

## 1. Summary
Sentences from Indian listed companies' annual reports, with section and page information. Labelled for whether a sentence states a business risk, its category, and whether it is a **specific red flag** or **generic boilerplate**. The dataset is built for an early-warning tool for credit analysts.

| | |
|---|---|
| Reports | **26** (10 distressed, 16 healthy) from 25 companies |
| Pages | 8,303 (1.8% of pages have no text layer) |
| Sentences after cleaning | **96,792** (median 20 words) |
| Labelled so far | 300 (pilot) |
| Language | English |
| Years | FY2018–FY2020 (+ Gensol FY2024 as a backtest case) |

## 2. Motivation
- **Task:** sentence-level risk detection → category → specificity, for evidence-cited red-flag reports.
- **Gap:** existing risk datasets are US 10-K "Item 1A" (a dedicated risk section). Indian reports have no single risk section; red flags sit in the auditor's report, the CARO annexure, notes on contingent liabilities, and the secretarial audit. No public, labelled Indian dataset was found.
- **Users:** credit analysts at banks and NBFCs (RBI early-warning-signal monitoring).

## 3. Composition

### Companies and groups
| Group | Companies (fiscal year) | Event used to define "distressed" (to be verified individually) |
|---|---|---|
| **Distressed (10 reports)** | Jet Airways (FY18), DHFL (FY18, FY19), Reliance Capital (FY20), Reliance Communications (FY18), Coffee Day Enterprises (FY19), IL&FS Transportation (FY18), Jaiprakash Associates (FY19), Yes Bank (FY19), Gensol Engineering (FY24) | Default, insolvency (IBC), RBI action or SEBI order within ~2 years after the report |
| **Healthy (16)** | Asian Paints, Bajaj Finance, Dr Reddy's, Havells, HUL, Infosys, L&T, Maruti, NTPC, Pidilite, SRF, Sun Pharma, Tata Steel, TCS, Titan, UltraTech (all FY19) | No such event; large, stable companies (survivorship bias, see §7) |

Distressed and healthy reports come from the **same years (FY2018–20)**, so differences aren't explained by changes in reporting rules over time.

### Splits (by company, no company in two splits)
| Split | Companies | Reports | Sentences |
|---|---|---|---|
| Train | 16 (5 distressed, 11 healthy) | 17 | 66,879 |
| Validation | 3 (RCOM, Titan, NTPC) | 3 | 10,688 |
| Test | 5 (Jet Airways, Yes Bank, TCS, Maruti, Sun Pharma) | 5 | 15,830 |
| Backtest (demo only, never trained on) | 1 (Gensol) | 1 | 3,395 |

### Sentences by detected section
| Section | Sentences | | Section | Sentences |
|---|---|---|---|---|
| Notes to accounts | 39,102 | | Auditor's report | 5,365 |
| Board's report | 18,260 | | CARO annexure | 2,082 |
| MD&A | 13,590 | | Secretarial audit | 1,655 |
| Corporate governance | 10,374 | | Contingent-liability notes | 1,272 |
| Business responsibility | 4,926 | | IFC report | 166 |

Excluded: cover/contents pages, AGM notice, and the primary financial statement tables.

### Record format (`data/interim/sentences.jsonl`)
`sentence_id, company, fy, group, page, section, scope (standalone/consolidated), text, n_words, has_money, has_date, kw_hit`

Labels (`data/annotations/*.jsonl`): `is_risk (0/1), category (Financial|Legal|Regulatory|Operational|Business|None), category_2, specificity (Specific|Generic|None), notes, annotator`

## 4. Collection
- **Source:** public annual-report PDFs from the BSE archive (`bseindia.com/bseplus/AnnualReport/<scrip>/…`) and a BSE corporate filing (Gensol). URL, download date, SHA-256 and page count for each file are in `data/raw/manifest.csv`.
- **Method:** normal HTTP downloads of public files, rate-limited (1 request/second). The NSE API refuses automated requests, so it wasn't used.
- **Date:** 2026-10-28.

## 5. Preprocessing (`src/prep/`)
1. `01_ingest.py`: manifest, checksum, text-layer check (all 26 passed).
2. `02_extract.py` (PyMuPDF):
   - **Rupee fix:** older reports draw ₹ with a custom font, which comes out as `` ` `` or `H`. Text in any "Rupee" font becomes ₹.
   - Removal of running headers and footers; joining of hyphenated line breaks; Unicode normalisation; flagging of table-like blocks.
   - **Section detection:** headings (large or bold text, running headers) plus standard CARO/IFC wording. The auditor's report was found in 26/26 reports and CARO in 23/26.
3. `03_sentences.py`:
   - Joins blocks split mid-sentence.
   - pysbd sentence splitting protected for Indian abbreviations (Rs., Ltd., Pvt., No., viz.).
   - Repairs broken "fi"/"fl" ligatures ("fi led" → "filed") and removes control characters.
   - Keeps sentences of 6–120 words that are mostly letters.
   - Removes **exact duplicates within a report** (standalone vs consolidated notes repeat each other): 16,715 removed.
4. `04_split_sample.py`: company-wise split and stratified pilot sample.

## 6. Labelling
- **Guide:** `data/annotations/guide.md` (v1.1). Decision rules for each category, tie-breakers, and Specific vs Generic.
- **Pilot (done):** 300 sentences from train/val companies, sampled 40% from high-signal sections, 20% by keyword match and 40% at random. Labelled by an LLM (Claude) applying the guide. These labels are **pre-labels to be reviewed by humans**, not gold labels.
- **Agreement (in progress):** 100 pilot sentences exported blind (`pilot_100_blind_for_humans.xlsx`) for team members to label independently. `06_agreement.py` then computes Cohen's kappa. Target ≥ 0.6.
- **Full plan (M3):** ~2,200 sentences. Train/val are pre-labelled by the LLM, then corrected by humans. The **test set is labelled by humans only**, to avoid anchoring on LLM suggestions.

### Pilot results (n = 300)
| Measure | Value |
|---|---|
| Risk sentences | 36 (12.0%); **imbalance 7.3 : 1** |
| Risk rate by stratum | random 5.0% · high-signal sections 14.2% · keyword 21.7% |
| Risk rate by group | **distressed 23.5% vs healthy 6.4%** (χ² = 16.6, p < 0.001) |
| Categories | Financial 14 · Legal 9 · Regulatory 7 · Operational 4 · Business 2 |
| Specificity | Specific 32 · Generic 4 (high-signal sections are mostly specific) |
| Extraction errors | 22 / 300 (7.3%), mostly table fragments → next cleaning target |

## 7. Statistics and known issues
- **Keyword rules are not enough** (`reports/figures/04`):
  - "Going concern" and "default" wording appears in **100% of healthy reports**, because standard auditor text uses these words.
  - Only "qualified/adverse opinion" separates the groups cleanly (40% of distressed vs 0% of healthy).
  - Red-flag keyword density is higher in distressed reports (median 12.7 vs 5.3 per 1,000 sentences, Mann-Whitney one-sided p = 0.016), but the groups overlap.
  
  This is the reason for a trained classifier with a Specific/Generic label.
- **Year-over-year copying:** 27.7% of DHFL FY2019 sentences are copied word for word from FY2018. This supports the YoY-diff feature, and it is why splits are by company.
- **Company-specific vocabulary:** company names (e.g. "jaypee", "reliance") are among the words most typical of distressed auditor reports. A model could learn names instead of risk language, so company-wise splits are required. 21.3% of test vocabulary is unseen in training.
- **Biases:**
  - The healthy group is large-cap (survivorship bias).
  - 10 distressed reports is a small sample.
  - English only.
  - Text-layer PDFs only; scanned reports are not covered.
  - Sector mix is uneven: finance and infrastructure are heavier in the distressed group.
- **Section detection misses:** CARO not found in 3/26 reports; HUL's combined Board's/MD&A report is mis-tagged; section headings sometimes merge into the next sentence.

## 8. Privacy and ethics
- All sources are **public statutory filings**. No personal data beyond what companies publish (names of directors, auditors and officials).
- The tool **profiles companies, not people**. Person names are kept only as entities within evidence sentences, and no person-level scoring is done.
- Under India's DPDP Act 2023, personal data a person has made public (or that is published under a legal obligation) falls under an exemption. We still minimise it: no separate person index.
- Labels describe **disclosed risk language**, not a judgement of fraud. Outputs are worded as "review recommended".
- Production deployments should keep borrower documents on the bank's own infrastructure (the classifiers run on CPU; the LLM is optional or local).

## 9. Licence and use
- Annual reports are public disclosures under SEBI LODR; they are used here for non-commercial academic research. Raw PDFs are **not redistributed** (git-ignored); the manifest lists their sources.
- Labels and code: for academic use within the course.

## 10. Maintenance
Version tags: `v0.1` = pilot (this card); `v1.0` = gold labels frozen at the end of M3 week 2. Contact: project team.
