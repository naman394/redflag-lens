# Product Spec: What to Build (Industry Track)

*Working name: **RedFlag Lens**, an early-warning copilot for annual reports of Indian listed companies. Written 2026-10-27. Builds on [RESEARCH.md](RESEARCH.md) and [MILESTONES.md](MILESTONES.md).*

---

## 1. The pitch in three lines

**For** credit analysts at Indian banks and NBFCs **who** must review a borrower's 200–400 page annual report for warning signs, **RedFlag Lens** reads the report in minutes. It surfaces red flags with the exact page and sentence, shows **what changed since last year's report**, and drafts an evidence-backed credit-note section the analyst can verify in one click.

Why this beats "a risk classifier with a dashboard": it is anchored to a **regulatory requirement banks already have** (RBI's early-warning-signal framework), it uses a **finding from finance research** (changes in report language carry signal), and it can be **validated on real defaults**.

---

## 2. Why this problem, why now

### 2.1 Banks are required to do this
- RBI's **Master Directions on Fraud Risk Management** (July 2024, effective 15 July 2024) require commercial banks and all-India financial institutions to run a board-approved **Early Warning Signals (EWS) and Red-Flagging of Accounts** framework. EWS must be integrated with core banking systems, and **data-analytics and market-intelligence units are mandatory**.
- After *SBI v. Rajesh Agarwal* (2023), banks must give borrowers notice and a hearing before classifying accounts as fraud. So **each flag needs documented evidence**, which is exactly what a sentence + page citation gives.
- **Product consequence:** frame the output as *EWS indicators with evidence*, not "AI risk scores". Use escalation language ("review recommended"), never verdicts ("fraud").

### 2.2 Text predicts distress
- Mayew, Sethuraman & Venkatachalam (2015): going-concern language and MD&A tone predict bankruptcy, **incrementally to financial ratios and even the auditor's going-concern opinion, up to three years ahead**.
- Mai et al. (2019), Matin et al. (2019, Danish auditor reports, AUC ≈ 0.84), Hájek & Olej (2015): text features add predictive power. Bankrupt firms use more uncertain, modal and negative language.
- **"Lazy Prices"** (Cohen, Malloy & Nguyen, *Journal of Finance* 2020): firms mostly copy last year's report, and **when they change risk language, it signals bad news** that markets are slow to price. This is the basis for the year-over-year diff feature.

### 2.3 Recent Indian cases show the cost
IL&FS (2018), DHFL (2019), Café Coffee Day (2019), Jet Airways (2019), SREI (2021), Reliance Capital (2021), Future Retail (2022), and **Gensol Engineering (2025)**, where SEBI's April 2025 interim order alleged ₹262 crore of EV-loan funds unaccounted for, round-tripping with a supplier, and forged no-default/conduct letters given to rating agencies. ICAI's review board is examining the FY2023-24 audit. These are the test cases for backtesting (§6).

### 2.4 Market gap
| Player | What they do | Gap |
|---|---|---|
| AlphaSense, Hebbia, Bloomberg | Global AI search across filings, cited answers; $3k+/seat/yr (Hebbia, third-party estimate) | General Q&A, US-centric, not EWS-shaped, expensive for Indian mid-size lenders |
| Precisa and similar Indian lending tools | Bank statements, GST and ITR analysis for NBFCs | No annual-report narrative analysis |
| InstaFinancials, Altinfo, Tofler-type data providers | MCA financials, director data, some red-flag reports | Mostly structured numbers, not text evidence from the report |
| Manual practice | Analysts reading PDFs + Ctrl+F + personal checklists | Slow, inconsistent, misses YoY changes |

No Indian product turned up that does **evidence-cited, red-flag extraction + YoY disclosure change on annual reports** for lenders. That gap is the positioning. (Search coverage was limited; say "we found no direct competitor", not "none exists".)

Signal that regulators care: NFRA and MeitY (IndiaAI) ran a financial and regulatory compliance AI challenge with a ₹1.5 crore prize pool (applications closed March 2026).

---

## 3. Users and their job

| | Primary: **Credit analyst** (bank/NBFC corporate credit, annual review) | Secondary |
|---|---|---|
| Job | Annual review of borrower; prepare the credit note; raise EWS flags | Equity analysts (same job, pre-investment); internal audit / compliance; retail investors |
| Pain | 4–8 hours per report (assumption to validate by interviewing 2–3 analysts); red flags buried in notes and annexures; easy to miss what changed vs last year | |
| Must have | Nothing important missed; every claim verifiable; output pastes into a credit note | |
| Won't accept | Black-box scores; hallucinated numbers; uploading confidential data to unknown services (annual reports are public, but their internal notes aren't) | |

**User validation for M1:** talk to 2–3 people who do credit or equity analysis (CA friends, bank contacts). Ask how long a review takes, what they check, and what they'd pay for. Quote them in the business presentation. This is the cheapest way to make the industry case credible.

---

## 4. The product: features

### MVP (must ship; maps to M3–M5)

**F1. Red-flag extraction with evidence** *(core NLP: your classifiers + NER/IE)*
- Sentence-level risk detection → category (Financial / Legal / Regulatory / Operational / Business) → **Specificity** (Generic boilerplate vs Specific event).
- Extracted fields: amount (normalised ₹), counterparty, date/FY, event type, section, page, bounding box.
- Ranking: Specific > Generic, by section weight (auditor/CARO > notes > MD&A) and by amount.

**F2. Red-flag checklist** *(rules + NLP; what analysts recognise immediately)*
A fixed checklist of ~20 indicators, each marked **Found / Not found / Unclear** with evidence links. Starting list, drawn from the auditor's report, CARO and governance disclosures (check exact clauses against current MCA/SEBI texts):

| # | Indicator | Where |
|---|---|---|
| 1 | Qualified / adverse opinion or disclaimer | Auditor's report |
| 2 | Material uncertainty related to going concern | Auditor's report |
| 3 | Emphasis of matter | Auditor's report |
| 4 | Default in repayment of loans or interest | CARO |
| 5 | Undisputed statutory dues outstanding > 6 months | CARO |
| 6 | Disputed statutory dues (amount, forum) | CARO |
| 7 | Fraud reported by/on the company | CARO / Board's report |
| 8 | Funds raised for short-term used for long-term | CARO |
| 9 | Cash losses in current/previous year | CARO |
| 10 | Resignation of statutory auditor | CARO / Board's report |
| 11 | Contingent liabilities large relative to net worth | Notes |
| 12 | Large loans/guarantees to related parties | Notes |
| 13 | Penalties/strictures by SEBI/exchanges (last 3 yrs) | Corporate governance report |
| 14 | Non-compliance observations | Secretarial audit (MR-3) |
| 15 | Pledged promoter shares (if disclosed) | Shareholding / governance |
| 16 | Change in auditors / KMP resignations | Board's report |
| 17 | Impairment / write-offs / erosion of net worth | Notes / MD&A |
| 18 | Restructuring / one-time settlement / NCLT proceedings | Notes / Board's report |
| 19 | Credit rating downgrade mentioned | Board's report / MD&A |
| 20 | Key customer/supplier concentration | MD&A |

**F3. Year-over-year disclosure diff** *(semantic similarity used where it matters; the "Lazy Prices" feature)*
- Upload this year's and last year's report. Align risk sentences across years with embeddings (+ TF-IDF baseline).
- Show findings that are **New**, **Removed**, **Escalated** (same topic, larger amount or stronger language), or **Unchanged**.
- New risks in the auditor's report or notes sit at the top. This is the feature that makes analysts say "I'd use that".

**F4. Evidence-grounded credit-note draft** *(the agentic layer)*
- The Report agent writes a 1-page summary in a credit-note format: key red flags, changes vs last year, items needing borrower clarification.
- **Every sentence cites finding IDs.** A Verifier (MiniCheck/NLI + exact ₹/date matching) removes unsupported claims.
- Output: PDF/DOCX + JSON (fields shaped for an EWS system).

**F5. Evidence viewer**
- Click any finding → the PDF page opens with the sentence highlighted. Analyst can mark 👍/👎 (feedback becomes new training data).

### Stretch (only if MVP is solid)
- **S1. Ask-the-report:** Q&A over the report with citations ("What is the total disputed tax demand?").
- **S2. Table extraction** for the contingent-liabilities note and the CARO disputed-dues table (Docling), giving exact totals.
- **S3. Peer comparison:** flag counts vs sector peers.
- **S4. Watchlist:** a portfolio of borrowers re-analysed automatically when new reports are filed.

**Explicitly out of scope:** fraud verdicts, stock recommendations, credit scores, non-English reports, scanned PDFs.

---

## 5. Architecture (industry-ready, still student-sized)

```
             ┌─────────────── Web UI (Streamlit MVP → optional React later) ──────────────┐
             │ upload 1–2 PDFs · dashboard · checklist · YoY diff · evidence viewer · memo │
             └───────────────────────────────┬─────────────────────────────────────────────┘
                                             │ REST
                                   ┌─────────▼─────────┐
                                   │   FastAPI (API)   │  /analyze  /compare  /report/{id}  /feedback  /health
                                   └─────────┬─────────┘
                                             │ job queue (FastAPI BackgroundTasks or RQ; no Redis needed for MVP)
     ┌───────────────────────────────────────▼───────────────────────────────────────┐
     │ Pipeline worker                                                               │
     │ 1 Ingest: PyMuPDF text+bbox, header/footer strip, ₹ fix, section detection    │
     │ 2 Segment: PySBD + abbreviation rules                                         │
     │ 3 Classify: risk → category → specificity (best model from M3/M4; ONNX on CPU)│
     │ 4 Extract: spaCy + EntityRuler (regulators, ₹ crore/lakh, FY dates), events   │
     │ 5 Checklist rules over sections + classifier outputs                          │
     │ 6 Embed + align (YoY diff), related findings                                  │
     │ 7 Agents: Document → Risk (tools: search_report, get_section) → Report        │
     │ 8 Verifier: MiniCheck + number/date match → drop/rewrite unsupported claims   │
     └───────────────────────────────────────┬───────────────────────────────────────┘
                                             │
                     SQLite/Postgres (jobs, findings, feedback) + local file store (PDFs)
                     Logs/metrics → monitoring dashboard (Streamlit page or Grafana)
```

| Concern | Choice | Why |
|---|---|---|
| Classifier serving | Best fine-tuned encoder exported to **ONNX**, CPU inference | Cheap to deploy; no GPU needed in production |
| LLM | Configurable: hosted API **or** local 7–8B model via Ollama | Banks prefer on-prem; this is a strong talking point |
| Agents | Plain Python + Pydantic schemas (LangGraph optional) | Easy to test each step; no framework lock-in |
| Storage | SQLite for demo, Postgres-ready via SQLAlchemy | |
| Packaging | Docker Compose (api + ui) | One-command demo |
| Deploy | Hugging Face Spaces / Render / small VM | Free or cheap |
| CI | GitHub Actions: lint, unit tests, small regression eval on 50 gold sentences | Shows engineering maturity; catches model regressions |
| Monitoring | Per-job metrics (stage latency, findings per category, mean confidence, verifier reject rate, LLM failures, 👍/👎 rate) + drift check vs test-set baseline (Evidently or a simple KS test) | Covers the M5 "monitoring strategy" requirement |
| Failure handling | No text layer → clear error; encrypted/corrupt → reject; >600 pages → limit; LLM down → template memo from structured findings; verifier reject → drop + flag; zero findings → explicit "no indicators found" | Covers the M5 "failure handling" requirement |

---

## 6. Validation: the backtest (your strongest demo)

**Question:** would RedFlag Lens have raised flags **in the 1–2 annual reports before** a company's default or regulator action, more often than for healthy peers?

- **Distressed set (~10–15 firms):** candidates are IL&FS group entities, DHFL, Reliance Capital, SREI, Future Retail, Café Coffee Day, Jet Airways, Sintex, Gensol, plus others from NCLT admissions. Verify each event and date from news or SEBI/NCLT records before use.
- **Control set:** same number of healthy firms, matched by sector, size and year.
- **Metrics:** red-flag count and checklist hits, distressed vs control (Mann–Whitney U test); ROC-AUC of "flag score" separating the two groups; lead time (how many years before the event the first Specific flag appeared).
- Keep these companies **out of the training data**.
- Be careful with the claim: the result is "flags were present and surfaced", **not** "the system predicts defaults". The sample is small; state that as a threat to validity.

Live demo script (M6): upload Gensol's (or another case's) report for the year before the regulator action → show the flags → show the YoY diff vs the previous year → open the generated memo with highlighted evidence.

---

## 7. Success metrics (set in M1, before building)

| Level | Metric | Target |
|---|---|---|
| Model | Risk-sentence recall / precision | ≥ 0.85 / ≥ 0.70 |
| Model | Category macro-F1 | ≥ 0.75 |
| Model | Specificity F1 | ≥ 0.80 |
| IE | ₹ amount value accuracy · ORG F1 | ≥ 0.90 · ≥ 0.80 |
| Checklist | Indicator accuracy vs manual review on 5 reports | ≥ 0.85 |
| YoY diff | Precision of "New/Escalated" items | ≥ 0.75 |
| Memo | Claims supported by evidence (verifier + manual check) | ≥ 95%, zero wrong ₹ amounts |
| System | End-to-end time per report on CPU | < 5 min |
| System | Cost per report (LLM) | < ₹20 (or ₹0 local) |
| Business | Analyst review time with vs without tool (small test, 2–3 people, 1 report each) | ≥ 50% reduction |
| Business | Backtest: distressed vs control separation | ROC-AUC ≥ 0.75 |

These are starting targets. Adjust them after the baseline, but **record the originals** for the M6 report.

---

## 8. Business case (for the M6 presentation)

- **Value:** analyst time saved per report × reports per year. Example with assumed numbers (validate with interviews): 5 hours saved × ₹1,500/hour loaded cost × 500 annual reviews = **₹37.5 lakh/year** for one mid-size NBFC credit team. The bigger value is one avoided bad loan, which is often worth crores.
- **Pricing hypothesis:** per-report (₹500–₹2,000) for small lenders and investors; annual seat or on-prem licence for banks.
- **Go-to-market:** NBFC and co-operative bank credit teams (underserved by expensive global tools) → rating-agency analysts → banks via on-prem deployment.
- **Moat:** an Indian-specific labelled dataset + red-flag checklist + feedback loop from analyst 👍/👎.
- **Risks:** liability for missed flags (mitigate: decision-support positioning, human in the loop); PDF variety; LLM costs (mitigate: local model + cheap encoders for most of the pipeline).

---

## 9. Build plan against the milestones

| Milestone | Build | Demo-able artefact |
|---|---|---|
| **M1** Problem | Problem brief, user interviews, checklist v1, success metrics table, feasibility | 2-page brief + interview notes |
| **M2** Data | Collect ~40 reports (+ prior-year reports for 10 of them for YoY; + backtest set), extraction pipeline, annotation (risk/category/specificity), data card, EDA, company-wise split | Data card + EDA notebook |
| **M3** Models | TF-IDF+LR baseline, keyword baseline, FinBERT/DeBERTa/SetFit, LLM zero-shot; experiment log; repo + CI | Model comparison table |
| **M4** Eval & optimise | Threshold tuning for recall, ONNX export, latency tests, error analysis, robustness suite (unseen sectors, ₹ variants, negation, noise), checklist + YoY diff evaluation, **backtest** | Evaluation report |
| **M5** Deploy | FastAPI + Streamlit, Docker, hosted demo, monitoring page, failure handling, agents + verifier with ablation | Live URL |
| **M6** Deliver | Technical report, README, business PPT, recorded backup demo | Final package |

**Team split (if 3–4 people):** (1) PDF ingestion + IE + checklist rules, (2) data/annotation + classifiers, (3) embeddings + YoY diff + agents/verifier, (4) API/UI/deploy/monitoring + business case. Everyone annotates.

---

## 10. What makes this stand out to evaluators
1. **Real regulatory hook** (RBI EWS 2024), not a made-up need.
2. **A finding nobody else in the class will have:** the backtest on real Indian defaults.
3. **YoY diff** grounded in published finance research ("Lazy Prices").
4. **Trust by design:** sentence-level citations, verifier, no verdict language, on-prem LLM option.
5. **Engineering maturity:** CI with a regression eval, ONNX CPU serving, monitoring, failure handling.

---

## Sources
- RBI Master Directions on Fraud Risk Management (2024) — https://rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=12702 · summary: https://www.grantthornton.in/globalassets/1.-member-firms/india/assets/pdfs/services/rbi-master-directions-on-fraud-risk-management.pdf · https://elplaw.in/wp-content/uploads/2024/07/RBI-Revised-Master-Directions-on-Fraud-Risk-Management-July-2024.pdf
- Borrower hearing before fraud tagging — https://outlookmoney.com/banking/banks-can-only-tag-accounts-as-fraudulent-after-seeking-borrowers-response-rbi
- Text and bankruptcy prediction (survey and studies) — https://arxiv.org/abs/2101.00719 · https://arxiv.org/abs/2208.11334 · https://arxiv.org/abs/2101.04480 · https://www.fuqua.duke.edu/node/173
- Lazy Prices — Cohen, Malloy & Nguyen, *Journal of Finance* 75(3), 2020 (cited from prior knowledge; not retrieved in this search)
- Gensol / SEBI interim order — https://www.business-standard.com/markets/capital-market-news/gensol-engg-slumps-as-sebi-unplugs-promoters-over-alleged-fraud-125041700568_1.html · https://www.taxmann.com/post/blog/analysis-sebi-and-icai-probe-gensol/ · https://www.angelone.in/news/sebi-cracks-down-on-gensol-engineering-over-funds-misuse-halts-stock-split
- Competitors — https://www.alpha-sense.com/compare/alphasense-vs-hebbia/ · https://www.hebbia.com/blog/alphasense-competitors · https://sacra.com/research/700m-yr-sacra-for-public-markets/ · https://www.startuphub.ai/startups/precisa.md · https://support.instafinancials.com/portal/en/kb/articles/what-is-instafinancials-27-7-2022
- NFRA/MeitY AI challenge — https://aikosh.indiaai.gov.in/home/competitions/details/b5ebae1e-73d1-4ba8-baa1-21beaabd1f6d
- Indian red-flag reference — https://www.bharatilawhouse.com/product/bharats-red-flags-compendium-practical-case-studies-in-forensic-accounting-corporate-fraud-investigation-by-ca-kamal-garg

*Caveats: market sizes and pricing are third-party estimates; the time-saved and cost numbers are assumptions to confirm through interviews; the checklist items and CARO clause wording must be checked against current regulatory texts; default/event dates for backtest companies must be verified individually.*

---

## Why not just use ChatGPT / an LLM? (viva answer)

**Framing:** we don't compete with LLMs. An LLM is one component (summary and verification). The product is the system around it. A frontier LLM may match or beat our classifier on single sentences, and we report that honestly as a baseline.

| Problem with "paste the report into an LLM" | What RedFlag Lens does |
|---|---|
| **Misses things.** A 300-page report is 150k+ tokens; long-context models summarise and overlook details buried mid-document ("lost in the middle"). | Checks **every sentence** in the auditor's report, CARO and notes. Recall is measured on a labelled test set. |
| **Unverifiable.** Invents amounts or page numbers. | Every finding links to the **exact sentence and page**. A verifier drops claims the evidence doesn't support. RBI's rules need documented evidence before an account is flagged. |
| **Inconsistent.** Different answers on different runs or prompts. | Same report → same findings. Fixed checklist, audit trail, model version logged. |
| **Data leaves the bank.** Borrower documents go to an external API. | Classifiers run **on-premises on CPU**; the LLM is optional and can be a local model. |
| **Costly at scale.** Every report runs through a large model. | Small fine-tuned models handle the bulk cheaply and quickly; the LLM sees only the shortlisted evidence. |
| **Generic.** No Indian specifics. | Understands ₹ lakh/crore and Indian number formats, CARO clauses, SEBI/NCLT; uses an RBI early-warning-style checklist. |
| **No structure across years.** | Year-over-year diff: new, removed and escalated risks. |
| **Unmeasured** on Indian annual reports. | Evaluated with precision, recall and F1 on a held-out Indian test set. |

**How we prove it (M4 experiment):** on 5 test reports, compare (a) the full report given to an LLM in one prompt vs (b) our pipeline, measuring: checklist recall, citation accuracy (does the cited page contain the claim?), number accuracy, consistency across 3 runs, cost per report and time per report. Also compare a zero-shot LLM vs our fine-tuned classifier on the sentence test set.
