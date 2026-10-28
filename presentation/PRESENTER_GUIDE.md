# RedFlag Lens: Presenter's Guide for Milestones 1 + 2

Everything you need for tomorrow, in plain language. Read it once fully, then use **Numbers to remember** and **Likely questions** as a quick revision before you present.

Slides: https://claude.ai/artifact/NPTo1j9XegdUq3HJ6ok13y

---

## 1. The project in one line

**We are building a tool that reads a company's annual report and finds the warning signs that the company may be in trouble, showing the exact sentence and page for each one.**

Name: **RedFlag Lens**. Track: **Industry Track**.

---

## 2. The problem

- Every listed company in India publishes an annual report. Ours average **319 pages**.
- When a bank lends money to a company, a **credit analyst** must read that report and judge whether the company is healthy.
- Warning signs are usually there, but **hidden in single sentences** among hundreds of pages. Examples:
  - "The auditor doubts the company can continue running" (going concern doubt)
  - "The company has not repaid ₹212 crore to banks" (default)
  - "SEBI imposed a penalty of ₹10 lakh" (regulatory action)
  - "A court case claiming ₹8.5 crore is pending" (litigation)
- Indian reports have **no single risk section** (unlike US reports). Warnings are spread across the auditor's report, the CARO annexure, the notes to accounts, the governance report and the secretarial audit.
- In our data, only **about 5% of random sentences** are risk statements, and the rest is normal text.

**Real example (from our own pipeline):** Jet Airways' FY2018 report, page 84. The auditor wrote that the company's survival depended on raising finance (a going-concern doubt). **A year later the airline was grounded.** The warning was public; someone just had to find it.

---

## 3. Who it's for, and why now

| | |
|---|---|
| **Main user** | Credit analysts at banks and NBFCs, checking a borrower's annual report before giving or renewing a loan |
| **Other users** | Risk / early-warning teams in banks, auditors, equity and bond analysts |
| **Why now** | RBI's **2024 rules on fraud risk management** require banks to run an "early warning signal" system. Before calling an account fraudulent, a bank must give the borrower a hearing, so every red flag needs **evidence**. |
| **Business objective** | Cut the first reading of a report from **hours to minutes**, miss fewer red flags, and link every finding to its page |

---

## 4. What we are building (how it works)

1. **Upload** the annual report (PDF).
2. **Extract** the text, fix the ₹ symbol, and work out which section each sentence comes from.
3. **Detect:** is this sentence a risk or not?
4. **Classify** into 5 types: **Financial** (debt, losses, defaults), **Legal** (court cases), **Regulatory** (SEBI, RBI, tax), **Operational** (plants, internal problems), **Business** (competition, demand).
5. **Specific vs Generic:** is it a *real red flag* ("we defaulted on ₹212 crore") or *boilerplate* ("we may face competition")? **This is our own addition and the most useful label.**
6. **Extract facts:** amount (₹ lakh/crore), date, company or regulator involved.
7. **Link:** find related sentences elsewhere in the report, and compare with last year's report (what's new or worse).
8. **Report:** a dashboard of red flags with the sentence and page. An AI writes a short summary using **only** the found sentences, and a checker removes any claim not backed by evidence.

**Simple picture:** a highlighter that reads all 300 pages, marks only the dangerous sentences, sorts them, and tells you the page number.

**NLP techniques used (what the course wants to see):**
- Traditional NLP: text cleaning, tokenisation, TF-IDF
- Machine learning: logistic regression (baseline)
- Transformers: FinBERT (a finance-trained BERT model)
- Information extraction: NER (names, amounts, dates)
- Embeddings: finding related sentences
- Generative / agentic AI: summary written only from evidence, plus a verifier

---

## 5. Why not just use ChatGPT? (they WILL ask)

Be honest: **on a single sentence, a good LLM may be as good as our model.** The difference is the whole system:

| Problem with "paste the report into ChatGPT" | What we do |
|---|---|
| Skims 300 pages and **misses details** in the middle | Checks **every sentence** in the important sections |
| Can **make up** amounts or page numbers | Shows the **real sentence and real page** |
| **Different answer** each time | **Same result** every time (needed for audits) |
| Bank documents go to an **outside company** | Small models run **inside the bank** |
| Expensive for thousands of reports | Fast and cheap |
| Not built for India | Understands ₹ lakh/crore, CARO, SEBI, NCLT |

We still **use** an LLM, only for the final summary, and only from evidence our models found.

In Milestone 4 we will **test** this: 5 reports, LLM vs our system, compared on red flags found, correct page numbers, consistency, cost and time.

**One line:** *"ChatGPT gives an opinion about a report. We give a complete, checkable list of warnings with page numbers, the same every time, that can run inside a bank."*

---

## 6. How we'll measure success (set before building)

| Level | What we measure | Target |
|---|---|---|
| Model | Finding risk sentences correctly (macro-F1) | ≥ 0.80 |
| Model | Catching real red flags (recall) | ≥ 0.85 |
| Model | Choosing the right category (macro-F1) | ≥ 0.70 |
| System | Cited page really contains the sentence | ≥ 98% |
| System | Summary claims backed by evidence | ≥ 95% |
| System | Time to process one report | ≤ 3 minutes |
| Business | Analyst's first review time | from [hours, to confirm by interview] to ≤ 30 min |
| Business | Troubled vs healthy companies told apart (ROC-AUC) | ≥ 0.75 |

**Why recall matters most:** missing a real red flag is worse than a false alarm the analyst can quickly dismiss.

---

## 7. Is it doable? (feasibility)

- **Data:** already collected, public and free.
- **Labelling:** about 2,200 sentences, roughly 25–35 person-hours. An AI suggests labels and people correct them; the test set is labelled by people only.
- **Computers:** a normal laptop for data work (the full pipeline runs in ~4 minutes), free Google Colab GPU for training.
- **Risks and fixes:**
  - Messy PDFs → cleaning steps (already done)
  - Categories overlap → written rules + an agreement check
  - Few troubled companies → add more in M3

---

## 8. What we did for Milestone 2

### 8.1 Collected the data
- **26 annual reports** from the BSE website (public).
- **10 from troubled companies**, using the report published 1–2 years before trouble: Jet Airways, DHFL (2 years), Reliance Capital, Reliance Communications, Coffee Day, IL&FS Transportation, Jaiprakash Associates, Yes Bank, Gensol.
- **16 from healthy companies:** Asian Paints, Bajaj Finance, Dr Reddy's, Havells, HUL, Infosys, L&T, Maruti, NTPC, Pidilite, SRF, Sun Pharma, Tata Steel, TCS, Titan, UltraTech.
- All from the **same years (FY2018–2020)**, so differences aren't due to rule changes over time. Gensol (FY2024) is the one recent case, kept aside for the demo.
- **8,303 pages** in total. Every file's source link, download date and fingerprint (checksum) is recorded.

### 8.2 Cleaned the data
Problems we actually found and fixed:

| Problem | Example | Fix |
|---|---|---|
| ₹ symbol came out wrong | "` 255.52 Crore", "H 53,301 Lakhs" | Old reports draw ₹ with a special font; we convert it back to ₹ |
| Broken words | "fi led", "classifi ed" | Joined back ("filed", "classified") |
| Sentences cut in half | "...the Company's ability" / "to raise finance..." | Joined when the next part starts in lowercase |
| Page headers and numbers mixed in | "Statutory Reports", "84" | Removed lines repeated on many pages |
| Same text printed twice | Standalone and consolidated notes | Removed **16,715** repeated sentences |
| Tables mixed with text | Rows of numbers | Flagged and kept out |

Result: **96,792 clean sentences**, each with its company, year, **section** and **page number**.

### 8.3 Found the sections
Our code detects which part of the report each sentence comes from. The auditor's report was found in **26 of 26** reports and the CARO annexure in **23 of 26**.

### 8.4 Split the data fairly
| Group | Companies | Sentences |
|---|---|---|
| Training (teach the model) | 16 | 66,879 |
| Validation (tune the model) | 3 | 10,688 |
| Test (final exam) | 5 | 15,830 |
| Backtest (demo only) | Gensol | 3,395 |

**Split by company, not by sentence.** Companies copy a lot of text year to year (DHFL repeated **27.7%** of its sentences word for word). If the same company appeared in both training and test, the model would "remember" it and the score would be falsely high.

### 8.5 Labelling rules (annotation guide)
Each sentence gets three labels:
1. **Risk?** yes / no
2. **Category:** Financial / Legal / Regulatory / Operational / Business
3. **Specific or Generic:** a real red flag, or boilerplate

The guide has clear rules for confusing cases. For example: a tax demand is **Regulatory**; a going-concern doubt is **Financial**; standard auditor wording is **not a risk**.

### 8.6 Pilot labelling (300 sentences)
We labelled 300 sample sentences to test the rules. **Important:** these are AI-produced first-draft labels following our guide, not final. The team is checking 100 of them independently to measure agreement (Cohen's kappa).

Results:
- Only **12%** are risk sentences (1 in 8); at random only **5%** (1 in 20). The data is very unbalanced, so we choose sentences smartly when labelling.
- Troubled companies: **23.5%** risk sentences. Healthy: **6.4%**. Statistically significant (p < 0.001).
- Most common types: Financial (14), Legal (9), Regulatory (7), Operational (4), Business (2).
- **7%** of sentences were broken table fragments, which is the next cleaning step.

### 8.7 Key findings from the analysis
1. **The riskiest sections are the smallest.** The auditor's report, CARO, contingent liabilities and secretarial audit together make up under 11% of the text but hold the strongest warnings.
2. **Troubled companies use more red-flag language.** A median of 12.7 red-flag sentences per 1,000 vs 5.3 for healthy companies (statistical test: p = 0.016).
3. **Keyword search is NOT enough** (our most important finding). "Going concern" appears in **all 16 healthy reports**, because auditors use it in standard wording. Only "qualified opinion" cleanly separates the groups (40% of troubled vs 0% of healthy). **This is why we need a trained model and the Specific/Generic label.**
4. **Company names leak.** Words like "jaypee" and "reliance" are among the most typical words in troubled reports, another reason to split by company.

### 8.8 Privacy and ethics
- Only **public** company filings.
- Director names appear in the text, but **we analyse companies, not people**: no person-level scores (in line with India's DPDP Act 2023).
- **No fraud verdicts.** The tool says "review recommended", and the human decides.
- Can run **inside the bank**, so documents don't go outside.

### 8.9 Known limitations (say these confidently; it shows maturity)
- Healthy companies are all large ones (survivorship bias).
- Only 10 troubled reports, a small sample.
- English only; scanned PDFs not supported.
- CARO not detected in 3 reports; about 7% table noise.
- Pilot labels are AI first drafts, not yet confirmed by people.

---

## 9. Have we trained a model?

**No, and that is correct for this stage.** M1 is planning and M2 is data preparation. Model training is **Milestone 3**.

| Milestone | What it needs | Status |
|---|---|---|
| M1 | Problem brief, users, objective, metrics, feasibility | ✅ Done |
| M2 | Data card, exploration, cleaning, statistics, split, privacy | ✅ Done |
| M3 | Baseline model + better models + experiment log | Next |

---

## 10. Next steps

**Next 2 weeks (finish data):**
- Team labels 100 sentences blind → agreement score (aim: kappa ≥ 0.6)
- Label ~2,200 sentences (AI suggests, people correct; test set labelled by people only)
- Remove table fragments
- Add more troubled companies
- Freeze the dataset as version 1.0

**M3: Models**
- Baseline: TF-IDF + logistic regression (simple word-based model)
- FinBERT (finance-trained transformer)
- SetFit (works well with few labels)
- Zero-shot LLM (GPT-style) as a comparison
- Optional: US company reports (SEC 10-K risk sections) as extra training data, plus a "trained on US, tested on India" experiment
- Track every experiment in Git

**M4: Evaluation**: accuracy, error analysis, robustness tests, and the **LLM vs our system** test.
**M5: Working app**: upload a PDF, get the red-flag dashboard (Streamlit/FastAPI), with monitoring.
**M6: Final**: report, GitHub repo, live demo, including the **Gensol backtest** (did its report show warnings before SEBI's 2025 action?).

---

## 11. Slide-by-slide script (19 slides)

1. **Cover:** "RedFlag Lens finds warning signs in Indian annual reports, with proof."
2. **Problem:** 319 pages, 5% risk sentences, no single risk section. The Jet Airways example.
3. **Users:** credit analysts; RBI 2024 rules require early-warning systems with evidence.
4. **Solution:** the 6-step pipeline; the LLM only sees evidence; a verifier checks it.
5. **Why not ChatGPT:** the comparison table; we'll test it in M4.
6. **Metrics:** model, system and business targets; recall matters most.
7. **Feasibility:** data done, labelling hours, compute, risks.
8. **M2 divider:** "Everything next is real data from our own pipeline."
9. **Data card:** 26 reports, 10 troubled vs 16 healthy, same years.
10. **Cleaning:** the ₹ font problem, broken words, 16,715 duplicates removed.
11. **Sections chart:** risky sections are small, so we oversample them.
12. **Density chart:** troubled reports have 2.4× more red-flag language (p = 0.016), but there is overlap.
13. **Keyword chart:** "going concern" is in every healthy report, so keywords fail and we need a model.
14. **Pilot:** 1 in 8 sentences are risks; 23.5% vs 6.4%; 7% table noise.
15. **Label schema:** real examples of the three labels.
16. **Splits:** by company; 27.7% copied year to year; company names leak.
17. **Privacy:** public data, companies not people, no fraud verdicts, runs inside the bank.
18. **Next steps:** done / next 2 weeks / M3 models.
19. **Close:** "The warnings were in the reports. We find them, and show where."

---

## 12. Numbers to remember

| Number | Meaning |
|---|---|
| **26** | annual reports (10 troubled, 16 healthy) |
| **8,303** | pages |
| **96,792** | clean sentences |
| **319** | average pages per report |
| **16,715** | duplicate sentences removed |
| **26/26, 23/26** | auditor's report / CARO sections found |
| **16 / 3 / 5 + 1** | companies in train / validation / test + backtest |
| **300** | pilot sentences labelled |
| **12% / 5%** | risk sentences in the pilot / at random |
| **23.5% vs 6.4%** | risk rate, troubled vs healthy (p < 0.001) |
| **12.7 vs 5.3** | red-flag sentences per 1,000, troubled vs healthy (p = 0.016) |
| **16/16** | healthy reports containing "going concern" wording |
| **27.7%** | DHFL sentences copied from the previous year |
| **~2,200** | sentences to label in total |

---

## 13. Likely questions and short answers

**Q: Why not just use ChatGPT?**
A: See section 5. It can miss details, invent numbers, varies run to run, and banks can't send documents outside. We use the LLM only for the summary, from evidence. We'll test both in M4.

**Q: Have you trained a model?**
A: Not yet. M1 and M2 are about the problem and the data; training is M3. Our data is ready and split.

**Q: Who labelled the 300 sentences?**
A: An AI applied our written guide to create first-draft labels. The team is checking 100 independently to measure agreement. The final test set will be labelled by people only.

**Q: Why only 26 reports?**
A: It's enough to build and test the pipeline, and it gives ~97,000 sentences. We'll add more troubled companies in M3. Labelling, not reports, is the real limit.

**Q: Why split by company?**
A: Companies copy text year to year (27.7% for DHFL), and company names are strong clues. Mixing them would give fake high scores.

**Q: How do you know which companies are "troubled"?**
A: Known public events within ~2 years of the report: loan defaults, insolvency cases, RBI action or a SEBI order. We'll verify each date with a source.

**Q: Why Indian reports and not US?**
A: The users are Indian banks. Indian reports have no single risk section, and no labelled Indian dataset exists, so ours is new. US data may be added as extra training data.

**Q: What's new in your project?**
A: (1) The first labelled sentence-level risk dataset for Indian annual reports. (2) The Specific vs Generic label (real red flags vs boilerplate). (3) Evidence-cited output with a verifier. (4) A backtest on real Indian company failures.

**Q: What if the model makes a mistake?**
A: It never gives a verdict, only "review recommended" with the sentence and page, so the analyst checks it in seconds. We focus on recall so real red flags aren't missed.

**Q: What's the business value?**
A: Faster report reviews, fewer missed warnings, and documented evidence for RBI compliance.

---

## 14. Before you present (checklist)

- [ ] Add team names and guide name on slide 1
- [ ] Replace "[hours]" on slide 6 (or say "to be confirmed through analyst interviews")
- [ ] (Optional) A teammate labels `data/annotations/pilot_100_blind_for_humans.xlsx`, then run `.venv/bin/python src/prep/06_agreement.py <file>` and add the kappa to slide 14
- [ ] Open the `RedFlagLens` folder on your laptop in case they ask to see code or data
- [ ] Revise **Numbers to remember** and **Likely questions**
