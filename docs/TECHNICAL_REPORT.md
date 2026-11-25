# RedFlag Lens: Technical Report

*NLP course project, Industry Track. Covers milestones M1 to M6.*
*Repository: this folder. Experiment log: `reports/experiments.csv`. Evaluation outputs: `reports/eval/`.*

---

## 1. Problem and users (M1)

Indian listed-company annual reports run 200–400 pages. The red flags a lender cares about are scattered through them:
- going-concern doubts, auditor qualifications and Emphasis of Matter paragraphs;
- unpaid or disputed statutory dues (CARO annex);
- loan defaults, insolvency (NCLT/IBC) proceedings, regulator orders and penalties;
- large contingent liabilities.

Screening one report by hand takes an analyst hours, and it is easy to miss something.

**Primary user:** credit analysts at banks and NBFCs doing lending decisions and annual reviews.
**Objective:** cut first-pass red-flag screening from hours to minutes, without missing material red flags, with every finding linked to its page and sentence so the analyst can check it.

**Targets set in M1, before building:**

| Level | Metric | Target | Result | Met? |
|---|---|---|---|---|
| Model | Risk-sentence detection F1 (test companies) | ≥ 0.80 | **0.86** (95% CI 0.76–0.93) on the 600-sentence test set, DeBERTa-v3-large ensemble (§4.7); FinBERT in the app: 0.77 | ✓ best model · ✗ app model |
| Model | Recall on risk sentences | ≥ 0.85 | **0.92** best model; 0.84 app model (600 sentences) | ✓ · borderline |
| Model | Category macro-F1 | ≥ 0.70 | 0.70 app model; 0.49 best model (600 sentences) | ✓ app · ✗ best |
| System | Findings whose cited page contains the sentence | ≥ 98% | by construction: the page is carried through from extraction (not separately measured) | ✓ (design) |
| System | Summary claims supported by evidence | ≥ 95% | not measured: the Gemini credit ran out before the LLM summary could be run; the demo uses the template summary, which quotes sentences directly | — |
| System | Time per report on a laptop CPU | ≤ 3 min | ~2.5 min (M2 MacBook Air, FinBERT) | ✓ |
| Business | Backtest separation, distressed vs healthy | ≥ 0.75 ROC-AUC | **0.94** on severe signs per 1,000 sentences (8 reports; raw flag count only 0.50), see §8 | ✓ (small sample) |

---

## 2. Data (M2)

- **26 annual reports** (10 from companies that later defaulted or collapsed, 16 healthy), 8,303 pages, downloaded from the BSE archive. Full data card: `data/DATA_CARD.md`.
- **Cleaning:** rupee-font repair, header/footer removal, ligature repair, abbreviation-safe sentence splitting, exact-duplicate removal. Result: **96,792 sentences**, each with section and page.
- **Fragment filter (new in M3):** `03b_flag_fragments.py` flags table rows, column headers and broken pieces: **14,684 sentences (15.2%)**.
  - On the 300 pilot labels it catches 81% of bad extractions and flags no risk sentences.
  - Rows are flagged, not deleted, so sentence IDs stay stable.
- **Splits are by company.** No company appears in two splits, because 27.7% of a company's sentences repeat word for word from the year before.
  - Train: 16 companies. Val: RCOM, TITAN, NTPC. Test: JETAIRWAYS, YESBANK, TCS, MARUTI, SUNPHARMA.
  - **GENSOL FY2024 is held out entirely**, as the backtest and demo case.

### 2.1 Labels (dataset v1.0)

No human labelling time was available, so labels come from several independent LLM annotators. The test set is graded only where annotators from different vendors agree.

| Split | Sentences | Risk | Labelled by |
|---|---|---|---|
| Train | 1,590 | 211 | Claude (300-sentence pilot) + `gemini-4-neon` (new sentences), guide v1.2 |
| Val | 260 | 30 | same |
| Test | 300 | 16 | **Claude** and **`gemini-4-argon`** independently; where they disagree (4 rows), `gemini-4-carbon` breaks the tie by majority |

- The training labeller (`gemini-4-neon`) **never labelled the test set**, so models are not graded against their own teacher.
- **Agreement:**

  | Comparison | is_risk κ | category κ | specificity κ |
  |---|---|---|---|
  | Claude vs gemini-4-argon (test set) | 0.904 | 0.874 | 0.905 |
  | Pilot (Claude) vs gemini-4-neon, 100 sentences | 0.942 | 0.888 | 0.942 |

  The pilot disagreements were settled by the team and led to guide v1.2.
- **Limitations of these labels:**
  - This is LLM–LLM agreement, not human gold labels.
  - Only **16 risk sentences** are in the test set, so every test score has a wide confidence interval. We report 95% bootstrap intervals.
  - Sampling for labelling was stratified: 35% from high-signal sections, 25% keyword hits, 40% random. This over-samples risk compared with a whole report.
  - About 350 sentences planned for train/val were not labelled because the Gemini credit ran out.

---

## 3. Models (M3)

All code is in `src/models/`. Random seeds are fixed, and every run is logged with its git commit in `reports/experiments.csv`.

| Model | What it is |
|---|---|
| **Keyword rules** | The Ctrl+F baseline: a 40-term red-flag regex, with category from the first matching keyword group |
| **TF-IDF + logistic regression** | Word 1–2-grams + character 3–5-grams, class-balanced; separate heads for risk, category and specificity |
| **Zero-shot open LLM** | Qwen2.5-7B-Instruct given only guide v1.2, 10 sentences per prompt, no training (RTX 4090) |
| **DeBERTa-v3 base / large** (§4.6) | Same 6-way head and inputs, 5 epochs (lr 3e-5 base, 1e-5 large), bf16 on an RTX 4090; `src/models/encoder.py` |
| **FinBERT, fine-tuned** | `ProsusAI/finbert` with one 6-way head (None + 5 categories), 4 epochs, learning rate 3e-5, softened class weights; risk score = 1 − P(None); 3 seeds on an RTX 4090 |

The section name is prepended to each sentence (e.g. `[CARO] …`), because the same words mean different things in the auditor's report and in the MD&A.

---

## 4. Results (M4)

Test set: 300 sentences from 5 companies never seen in training, 16 of them risk sentences.

| Model | Precision | Recall | F1 (95% CI) | Category macro-F1 | Speed |
|---|---|---|---|---|---|
| Keyword rules | 0.10 | 0.63 | 0.17 (0.08–0.27) | 0.15 | instant |
| TF-IDF + LR | 0.44 | 0.69 | 0.54 (0.32–0.71) | 0.41 | 0.2 ms/sentence |
| TF-IDF + LR, recall-tuned threshold | 0.14 | 0.94 | 0.24 | 0.27 | 0.2 ms/sentence |
| Zero-shot Qwen2.5-7B | 0.37 | 0.63 | 0.47 (0.27–0.64) | 0.43 | 126 ms/sentence on GPU (~6 min/report) |
| **FinBERT (seed 42, shipped)** | **0.64** | **0.875** | **0.74 (0.55–0.87)** | **0.61** | 0.7 ms GPU · 45 ms CPU |
| FinBERT, 3 seeds | 0.61 ± 0.04 | 0.85 ± 0.04 | **0.71 ± 0.04** | 0.55 ± 0.07 | |

**What this shows:**
1. **Keyword search is not enough.** It finds most risk sentences, but 9 in 10 of its flags are false alarms, because standard auditor wording ("going concern", "default") appears in every report.
2. **The fine-tuned FinBERT beats a 7-billion-parameter LLM** used zero-shot: F1 0.74 against 0.47. It is also about 180× faster per sentence and runs on a laptop CPU, so borrower documents never leave the bank. This answers "why not just ask ChatGPT?".
3. **Recall target met (0.875 ≥ 0.85).** The F1 and category targets were missed. With 16 test positives, the confidence interval on F1 (0.55–0.87) still includes the 0.80 target.

### 4.1 Threshold tuning

We tuned the threshold on the validation set (30 risk sentences) to keep recall ≥ 0.85.
- For FinBERT this produced a threshold of 0.05, which over-fits the small validation set: on test, precision falls from 0.64 to 0.39 with no gain in recall.
- **The product therefore uses two tiers:**
  - **High confidence:** score ≥ 0.5, shown by default.
  - **Review:** score ≥ the tuned threshold, behind a toggle.

  The analyst sees a short list first and can widen it when recall matters more.

### 4.2 Improvement iterations

| Version | Change | Effect |
|---|---|---|
| v1 | Keyword rules | test F1 0.17 |
| v2 | TF-IDF + LR, class-balanced | F1 0.54 |
| v3 | Section name added to the input | used by all later models |
| v4 | FinBERT fine-tuned (6-way head) | F1 0.74, recall 0.875 |
| v5 | Fragment filter in the pipeline | removes 15% of sentences that are table rows or headers |
| v6 | Nil-statement rule (`src/models/rules.py`) | removes CARO-style "no default / no pending litigation" sentences. Suppresses 47 train/val sentences and **0** true risks. No effect on the test set, which has almost no such sentences, but it removed the top false alarms on the Gensol report |
| v7 | Two-tier output (high / review) | short default list while keeping a recall-oriented tier |
| v8 | Template and boilerplate demotion (`is_template`, `models/boilerplate.txt`) | the first backtest (§8) showed healthy TCS and Maruti getting *more* flags per 1,000 sentences than Gensol and Yes Bank. Questionnaire rows ("Number of complaints…", "Is there any case filed…") and wording shared by ≥ 3 training companies (680 sentences) are moved to the review tier |
| v10 | Training augmentation (`src/models/augment.py`): 300 real nil statements + 300 template/boilerplate sentences from train companies, as non-risk. Synthetic negation ("has not received…") was tried and **dropped**: many rewrites stayed risky ("has not defaulted … and has incurred losses") | FinBERT 3-seed test F1 0.65 → 0.72; validation PR-AUC up for FinBERT, flat for DeBERTa (§4.6) |
| v11 | Encoder comparison: FinBERT vs DeBERTa-v3-base vs DeBERTa-v3-large, × {plain, augmented} × 3 seeds, seed ensembles; chosen on **validation** | DeBERTa-v3-large wins on validation (PR-AUC 0.92 vs 0.79) but ties the shipped FinBERT on test (F1 0.74); app keeps FinBERT (§4.6) |
| v9 | Severity tier (`is_severe`) | a fixed lexicon of early-warning signals (going concern, default or delay in repayment, audit qualification or adverse remark, insolvency/NCLT, fraud, Enforcement Directorate/SFIO, rating downgrade, pledge invocation, erosion of net worth, auditor resignation), written from RBI's early-warning-signal list **before** the backtest was re-run. Severe findings are listed first and lead the summary |

### 4.3 Error analysis (FinBERT, test set)

Of 300 sentences, 288 are fully correct.

| Error type | Count | Example / cause |
|---|---|---|
| False positive, other | 6 | descriptions of risk processes or exposure that are not a red flag |
| False positive, generic risk wording | 1 | "exposed to…" boilerplate |
| False positive, negated / nil statement | 1 | "no material…" wording; the pipeline's nil rule removes these |
| False negative, specific event missed | 2 | short headings of contingent-liability tables ("Cases pending before Appellate authorities…") |
| Category confusion | 2 | Legal ↔ Regulatory for disputes with a government party in court |

Per-sentence errors: `reports/eval/errors_finbert.csv`.

### 4.4 Robustness

At the recall-oriented threshold:

| Test | FinBERT | TF-IDF |
|---|---|---|
| Rupee symbol replaced by `Rs.` / `INR` / `` ` `` (14 sentences) | 1 prediction changed | 0 changed |
| 4% character-swap typos | recall unchanged; precision 0.39 → 0.30 | recall unchanged |
| **Negation:** 11 specific red flags rewritten as "there were no instances where…" | **9 still flagged by the model alone** | 9 still flagged |
| Unseen sectors (bank, airline, IT, auto, pharma) | recall 1.0 on JETAIRWAYS, MARUTI and SUNPHARMA; misses the single risk sentence of TCS and of YESBANK | |

- **Negation is the clearest weakness.** Both models react to risk words, not to whether the sentence denies them. The rule-based nil filter catches all 11 rewritten sentences, but the rule was written for exactly this pattern, so that result is optimistic.
- The fix is to add negated examples to the training data.

### 4.5 Inference cost

| | Time | Notes |
|---|---|---|
| FinBERT, RTX 4090 | 0.7 ms/sentence | |
| FinBERT, laptop CPU (M2) | 45 ms/sentence | ~2.3 min per median report of 3,038 sentences |
| Full pipeline, laptop CPU | ~2.5 min per report | text extraction adds ~10 s |
| Zero-shot 7B LLM | ~6.4 min per report | needs a 24 GB GPU |
| Hosted LLM API | — | would send ~300 requests per report, and the borrower's documents, to a third party |
| GPU used for training | ~$0.55 total | RunPod, 3 seeds × ~25 s plus setup |

---

### 4.6 Stronger encoders and seed ensembles (iterations v10–v11, on the original 300-sentence test set)

18 runs on one RTX 4090 (about 25 minutes, $0.47): 3 encoders × {plain, augmented} × 3 seeds. Each configuration's three seeds are averaged into an ensemble. **The rule, fixed in advance: choose by validation PR-AUC of the ensemble; look at test once, only to report.** Script: `src/models/ensemble.py`; log: `reports/model_comparison.csv`, `reports/encoder_grid_log.txt`.

| Configuration | Val PR-AUC (ens.) | Val F1 (ens.) | Test P | Test R | Test F1 (95% CI) | Test PR-AUC | Test cat. F1 |
|---|---|---|---|---|---|---|---|
| FinBERT, 3-seed ensemble | 0.79 | 0.68 | 0.61 | 0.69 | 0.65 (0.43–0.81) | 0.69 | 0.43 |
| FinBERT + aug | 0.84 | 0.75 | 0.65 | 0.81 | 0.72 (0.53–0.87) | 0.73 | 0.51 |
| DeBERTa-v3-base | 0.88 | 0.81 | 0.62 | 0.81 | 0.70 (0.51–0.85) | 0.68 | 0.52 |
| DeBERTa-v3-base + aug | 0.88 | 0.82 | 0.67 | 0.75 | 0.71 | 0.65 | 0.47 |
| **DeBERTa-v3-large** (selected) | **0.92** | 0.78 | 0.68 | 0.81 | **0.74 (0.55–0.88)** | 0.73 | 0.62 |
| DeBERTa-v3-large + aug | 0.90 | 0.79 | 0.70 | 0.875 | 0.78 (0.59–0.91) | 0.78 | 0.62 |
| *FinBERT seed 42 (shipped in the app)* | — | — | 0.64 | 0.875 | 0.74 (0.56–0.88) | 0.73 | 0.61 |

All test numbers use the 0.5 threshold. Thresholds tuned on validation for recall ≥ 0.85 all land at 0.05–0.07 and roughly halve precision, as for FinBERT in §4.1.

**Findings:**
1. **DeBERTa-v3-large is clearly better on validation.** PR-AUC is 0.92 against 0.79 for retrained FinBERT, and it is better on every seed.
2. **On the test set the gain disappears against the shipped model.** The selected DeBERTa-large ensemble scores F1 0.74, the same as the shipped FinBERT. Against retrained FinBERT seeds (ensemble F1 0.65), it is about +0.09. So the shipped FinBERT seed is on the lucky side of its own variance. With 16 test positives, the test set cannot separate models that are this close; all the intervals overlap.
3. **Augmentation helps on test** (FinBERT 0.65 → 0.72; DeBERTa-large 0.74 → 0.78) but not consistently on validation. By our rule we did not select the augmented model; we report it as promising.
4. **The F1 target of 0.80 is still not met on test.** The best validation F1s are 0.78–0.82.

**Decision at this point (superseded by §4.7):** the app keeps the shipped FinBERT. It scores the same on test, and it is much faster on a CPU: measured later at 45 ms against 341 ms per sentence for one DeBERTa-large (§4.7). That keeps a report under 3 minutes on a laptop CPU. DeBERTa-large is the upgrade path for a deployment with a GPU, where the ensemble takes about 20 seconds per report.

### 4.7 Test set v1.1: 600 sentences (iteration v12)

**Why.** With only 16 risk sentences, the original test set could not separate the models in §4.6; all the intervals overlapped.

**What we added.** 300 new sentences from the same 5 test companies, using the same stratified sampling with a different seed (`src/label/build_test_ext.py`). They were labelled with the same protocol as v1.0, using open models in place of Gemini because the Gemini credit had run out:
- **Annotator A:** Claude.
- **Annotator B:** Qwen2.5-32B-Instruct (AWQ, vLLM, temperature 0, JSON-constrained output; `src/label/open_llm_label.py`).
- **Tie-breaker C:** Llama-3.1-70B-Instruct (AWQ), a different model family, run only on the 12 disagreements.

| | Value |
|---|---|
| A vs B agreement on is_risk | 96.7% (κ = 0.755; lower than the 0.90 on v1.0) |
| Disagreements decided by C | 12 |
| Risk sentences in the extension | 21 |
| **Test set v1.1** | **600 sentences, 37 risk** (Legal 16, Financial 12, Regulatory 7, Business 2) |

Results are in `reports/label_agreement_v11.json`. Most disagreements were borderline generic-exposure sentences and auditor remarks.

**Re-scoring.** All 18 encoder runs (§4.6 grid) were retrained on an L40S and scored on all 600 sentences. Selection again used validation only: by validation ensemble PR-AUC, DeBERTa-v3-large + aug (0.878) and DeBERTa-v3-large (0.875) are effectively tied. In the §4.6 run the order was reversed, so the choice between the two is within noise. The baselines were re-scored on the new sentences with unchanged models and thresholds. Script: `src/eval/test_v11.py`; table: `reports/eval/test600_comparison.csv`.

**Test set v1.1, 600 sentences (37 risk), threshold 0.5:**

| Model | Precision | Recall | F1 (95% CI) | PR-AUC | Category macro-F1 |
|---|---|---|---|---|---|
| Keyword rules | 0.13 | 0.73 | 0.23 (0.15–0.30) | 0.11 | 0.18 |
| TF-IDF + LR | 0.52 | 0.70 | 0.60 (0.46–0.71) | 0.49 | 0.34 |
| Zero-shot Qwen2.5-7B | 0.45 | 0.73 | 0.56 (0.42–0.67) | 0.35 | 0.63 |
| FinBERT seed 42 (**in the app**) | 0.70 | 0.84 | 0.77 (0.65–0.86) | 0.80 | **0.70** |
| FinBERT + aug, 3-seed ensemble | 0.73 | 0.81 | 0.77 (0.66–0.86) | 0.82 | 0.59 |
| DeBERTa-v3-base + aug, 3-seed ensemble | 0.71 | 0.81 | 0.76 (0.64–0.86) | 0.83 | 0.36 |
| DeBERTa-v3-large, 3-seed ensemble | 0.73 | 0.86 | 0.79 (0.68–0.88) | 0.89 | 0.50 |
| **DeBERTa-v3-large + aug, 3-seed ensemble** | **0.81** | **0.92** | **0.86 (0.76–0.93)** | **0.89** | 0.49 |

**Single models (mean ± sd over 3 seeds):**

| Model | F1 | Recall | PR-AUC |
|---|---|---|---|
| DeBERTa-v3-large + aug | 0.84 ± 0.03 | 0.88 ± 0.03 | 0.83 ± 0.03 |
| DeBERTa-v3-large | 0.76 ± 0.03 | 0.84 ± 0.02 | 0.81 ± 0.05 |
| FinBERT | 0.75 ± 0.02 | 0.84 ± 0.02 | 0.77 ± 0.02 |

**On the 300 new sentences alone** (a fresh holdout that no choice was ever made on), DeBERTa-large + aug scores F1 0.93 (0.84–1.00) with recall 1.00, and FinBERT in the app scores 0.79.

**What this shows:**
1. **With 37 positives the models separate.** DeBERTa-v3-large with the extra non-risk examples is the best detector. It meets the F1 target (0.86 ≥ 0.80) and the recall target (0.92 ≥ 0.85), and its interval no longer reaches down to the baselines.
2. **The extra examples matter most for DeBERTa-large.** Single-model F1 rises from 0.76 to 0.84, mainly through precision: fewer false alarms on template and nil text.
3. **Category assignment is the weak point of the ensembles.** Their macro-F1 is 0.49 against 0.70 for the app's FinBERT. Detection improved; choosing between Legal and Regulatory did not. The macro average is also sensitive to the rare Business class, which has only 2 examples.
4. **Caveat on the labels.** The extension's labellers agree less (κ 0.76) than the v1.0 pair, and one of them is Claude, which also wrote the guide. The ranking of models is consistent across both halves of the test set, which supports it.

**Saved model (iteration v13).** The grid runs were not saved, so the selected configuration was retrained once (seed 42, A40 GPU) with `--save`. It is stored in fp16 as `models/debv3l_aug/` (849 MB) and loaded by the app with `REDFLAG_MODEL=deberta`. Its review threshold, tuned on validation for recall ≥ 0.85, is 0.05.

Scored on the laptop (CPU, float32), test v1.1, 600 sentences:

| | DeBERTa-v3-large + aug (saved) | FinBERT (app default) |
|---|---|---|
| Precision | 0.79 | 0.70 |
| Recall | **0.89** | 0.84 |
| F1 | **0.84** | 0.77 |
| PR-AUC | **0.90** | 0.80 |
| Category macro-F1 | 0.29 | **0.70** |
| Speed, laptop CPU | 341 ms/sentence, ~16.5 min per report | 45 ms/sentence, ~2.5 min per report |
| Speed, GPU (RTX 4090) | 1.4 ms/sentence, ~5 s per report | 0.3 ms/sentence |

**How it was checked:**
- No invalid (NaN) outputs. Scores differ from the GPU run by at most 0.07, from the fp16 conversion and a different library version.
- F1 0.84 matches the single-model mean in the table above.

**Decision.**
1. **The laptop app keeps FinBERT as the default.** DeBERTa-large is about 7.6× slower on a CPU, so a report takes about 16 minutes against the 3-minute target.
2. **The recommended production setup is DeBERTa-v3-large + aug on a GPU server:** about 5 s of classification per report, or about 15 s for the 3-seed ensemble (F1 0.86, recall 0.92).
3. **DeBERTa's category head is weak** (macro-F1 0.29; correct category on about half of detected risks). The next step is a hybrid: DeBERTa decides *whether* a sentence is a risk, and FinBERT assigns the category.
4. **Speed for laptops:** distilling DeBERTa into a smaller model is the main option.

### 4.8 Two-step cascade on CPU (iteration v14)

**Idea.** FinBERT is fast on a CPU but less accurate; DeBERTa-v3-large is more accurate but about 8× slower. In the cascade, FinBERT scores every sentence and supplies the category; only sentences with a FinBERT score ≥ 0.05 are re-scored by the saved DeBERTa model (`models/debv3l_aug`). The gate 0.05 is FinBERT's existing validation-tuned review threshold, so nothing was tuned on test. Code: `Cascade` in `src/app/pipeline.py` (`REDFLAG_MODEL=cascade`); evaluation: `src/eval/cascade.py`, `reports/eval/cascade.json`.

**Test set v1.1 (600 sentences, 37 risk), threshold 0.5, laptop CPU:**

| Setup | Precision | Recall | F1 | Risks lost at the gate |
|---|---|---|---|---|
| FinBERT alone (app default) | 0.70 | 0.84 | 0.77 | — |
| DeBERTa-v3-large + aug alone (saved single model) | 0.79 | 0.89 | 0.84 | — |
| **Cascade, gate 0.05** | **0.80** | **0.86** | **0.83** | 3 (of 37) |
| Cascade, gate 0.01 | 0.79 | 0.89 | 0.84 | 0 |
| Cascade, gate 0.1 / 0.2 | 0.80 | 0.84 | 0.82 | 4 |

**Cost on full reports (classification only, laptop):** the gate passes 7% of TCS's sentences and 12% of Jet Airways'. Classification takes 2.2 and 2.5 minutes, against 1.3–1.5 minutes for FinBERT alone and about 9–10 minutes for DeBERTa alone. With PDF extraction and rules, the whole pipeline took 3.2–4.3 minutes on the two fresh reports (§8.1), slightly above the 3-minute target.

**Decision.** The cascade gives almost all of DeBERTa's gain on a CPU. FinBERT stays the fastest option for a laptop; the cascade is the recommended CPU setup when about 4 minutes per report is acceptable, and DeBERTa on a GPU remains the fastest accurate setup.

## 5. Product (M5)

**Pipeline** (`src/app/pipeline.py`):

```
PDF ─► validate ─► extract text + sections ─► sentences ─► drop fragments ─► FinBERT ─► nil rule ─► template/boilerplate demotion ─► severity tier ─► ₹ amounts, dates
                                                                                                        │
                    dashboard / API JSON ◄── verifier ◄── LLM summary citing finding ids ◄──────────────┘
                                                  (template summary if the LLM is unavailable)
```

- **API** (`src/app/api.py`, FastAPI):
  - `POST /analyze` takes a PDF and returns JSON findings.
  - `POST /feedback` takes a 👍/👎 on a finding.
  - `GET /health` returns status.
  - Models load once at startup.
- **Dashboard** (`src/app/ui.py`, Streamlit):
  - upload a PDF or pick a sample;
  - severe warning signs first (⚠), then headline counts, verified summary, findings by category;
  - filterable findings table;
  - click a finding to see its **page with the sentence highlighted**;
  - 👍/👎 feedback on each finding.
- **Summary verifier:** the LLM must cite finding IDs for every claim. A claim is dropped if:
  - it cites an ID that doesn't exist, or
  - a ₹ amount in the claim does not appear in the cited sentences.

  The number of dropped claims is logged. The verifier is implemented but has not yet run on real LLM output, because the Gemini credit ran out.
- **Deployment:** a `Dockerfile` builds the API (port 8000) and dashboard (port 8501) with the models included. `GEMINI_API_KEY` is optional.

### 5.1 Failure handling

| Situation | Behaviour |
|---|---|
| Not a PDF / corrupt | 422: "could not be opened as a PDF" (tested) |
| Password-protected | 422: "password-protected, upload an unlocked copy" (tested) |
| Scanned (no text layer) | 422: "looks scanned, OCR it first" (tested) |
| > 700 pages or > 60 MB | 413 / 422 with the limit |
| LLM down, out of credit or bad output | falls back to a template summary built from the findings; error logged |
| LLM claim not supported by evidence | dropped by the verifier; count logged |
| No red flags found | says so explicitly, and reminds the analyst that this is not a guarantee |

### 5.2 Monitoring

`logs/reports.jsonl` gets one line per report, containing:
- page and sentence counts;
- findings per category, and the finding rate;
- mean risk score and the **share of low-confidence scores** (0.35–0.65);
- verifier rejections, summary source, LLM errors;
- time per stage.

Alert rules for production:
- the finding rate or low-confidence share moves more than 2× from the test-set baseline (a sign of a new report format or sector);
- LLM error rate above 10%.

`logs/feedback.jsonl` collects analyst votes, which become new labelled examples for the next training round.

### 5.3 Privacy

- Reports are public statutory filings. The tool profiles **companies, not people**.
- The classifier runs locally. Only the optional summary step sends text (the top 40 flagged sentences) to an LLM API, and an open-weights model can replace it on the bank's own servers.
- PDFs are not redistributed; the manifest lists their source URLs.

---

## 6. Limitations

1. **No human gold labels.** Labels are LLM-generated, with LLM–LLM agreement κ ≈ 0.90. Systematic errors that all models share would not show up.
2. **Test set still modest** (600 sentences, 37 risk, after the v1.1 extension): the F1 interval for the best model is 0.76–0.93; Business has only 2 test examples. The extension's labels have lower inter-annotator agreement (κ 0.76).
3. **Negation:** the model alone flags negated red flags; a hand-written rule patches the common CARO patterns.
4. **Tables:** about 830 rows from contingent-liability and disputed-dues tables contain real ₹ amounts but are excluded as fragments. A table extractor is future work.
5. **Corpus:** 26 reports, mostly large companies, English only, text-layer PDFs only.
6. **Agents:** the proposal's multi-agent design was reduced to a single LLM summariser plus a rule-based verifier because of time.

## 7. Future work

- Human-label the test set (about 2 hours) to confirm the LLM labels.
- Add negated and nil examples to training.
- Extract amounts from contingent-liability and CARO tables.
- Year-over-year diff: what is new in this year's report.
- More distressed companies.
- Retrain from analyst feedback.

## 8. Backtest

The full pipeline (`src/eval/backtest.py`, no LLM) was run on 8 complete reports from companies the model never trained on:
- **Gensol FY2024**, held out from everything (the company collapsed in 2025);
- the 4 test companies with a distressed or healthy label;
- the 3 validation companies, which were used only to choose the threshold.

The question: does the output tell a distressed company's report from a healthy one's?

**First run (v7, `reports/backtest/v1/`): no.** Healthy TCS (41.2) and Maruti (45.9) had *more* high-confidence flags per 1,000 sentences than distressed Gensol (35.5) and Yes Bank (36.6). Most of those TCS and Maruti flags were correct under the guide: the Epic lawsuit (₹6,499 crore), tax disputes, the CCI order. Large healthy companies disclose a lot of routine litigation, and questionnaire rows add noise on top. Counting flags measures how much a company discloses, not how distressed it is.

**Fix (v8 + v9, §4.2):**
1. Template and boilerplate text moves to the review tier.
2. A fixed, pre-registered severity lexicon based on RBI early-warning signals marks *severe* findings.

**Second run (`reports/backtest/summary.csv`):**

| Report | Group | Sentences | High-confidence flags | Flags / 1k | Severe | Severe / 1k |
|---|---|---|---|---|---|---|
| RCOM FY2018 | distressed (val) | 2,853 | 218 | 76.4 | 29 | **10.16** |
| Jet Airways FY2018 | distressed (test) | 1,954 | 117 | 59.9 | 9 | **4.61** |
| Yes Bank FY2019 | distressed (test) | 3,470 | 113 | 32.6 | 9 | **2.59** |
| Titan FY2019 | healthy (val) | 2,730 | 107 | 39.2 | 3 | 1.10 |
| Gensol FY2024 | distressed (held out) | 2,877 | 86 | 29.9 | 2 | **0.70** |
| NTPC FY2019 | healthy (val) | 3,476 | 129 | 37.1 | 2 | 0.58 |
| Maruti FY2019 | healthy (test) | 2,440 | 108 | 44.3 | 1 | 0.41 |
| TCS FY2019 | healthy (test) | 2,497 | 96 | 38.4 | 0 | 0.00 |

**Separation (ROC-AUC, distressed vs healthy, `reports/backtest/auc.json`):**

| Score per report | ROC-AUC (8 reports) | Gensol + test companies only (5) |
|---|---|---|
| High-confidence flags / 1k sentences | 0.50 | 0.33 |
| Specific flags / 1k sentences | 0.63 | — |
| **Severe signs / 1k sentences** | **0.94** | **1.00** |

**Third run, with the two-step cascade (§4.8; `REDFLAG_MODEL=cascade src/eval/backtest.py`, `reports/backtest/cascade/`):**

| Report | Group | Severe (FinBERT run) | Severe (cascade) | Severe / 1k (cascade) |
|---|---|---|---|---|
| RCOM FY2018 | distressed (val) | 29 | 35 | **12.27** |
| Jet Airways FY2018 | distressed (test) | 9 | 9 | **4.61** |
| Yes Bank FY2019 | distressed (test) | 9 | 7 | **2.02** |
| Titan FY2019 | healthy (val) | 3 | 3 | 1.10 |
| Gensol FY2024 | distressed (held out) | 2 | 3 | **1.04** |
| NTPC FY2019 | healthy (val) | 2 | 2 | 0.58 |
| Maruti FY2019 | healthy (test) | 1 | 0 | 0.00 |
| TCS FY2019 | healthy (test) | 0 | 0 | 0.00 |

- **Separation is unchanged:** severe signs / 1k give ROC-AUC 0.94 on all 8 reports and 1.00 on Gensol + test companies; the raw flag rate gives 0.56 and 0.33 (`reports/backtest/cascade/auc.json`). Gensol vs Titan (1.04 vs 1.10) is still the only misordered pair.
- **Cleaner severe findings:** Gensol's three are all real (p.100 discrepancies in stock statements submitted to the bank, now caught; p.184 going concern; p.187 adverse remark). Maruti's routine going-concern statement is gone. Yes Bank keeps four noisy matches (three share acquisitions caught by the pledge term, "downgraded World GDP growth") next to real ones (whistle-blower complaint, NPA classification).
- **Time:** 1.9–4.0 minutes per report on the laptop, against about 2.5 for FinBERT alone.

**Gensol, a year before collapse.** The two severe findings lead the summary:
- p.187: the subsidiary auditor's adverse remark on Scorpius Trackers;
- p.184: "conditions that may cast significant doubt on the Group's ability to continue as a going concern".

Other true findings appear in the high tier but not the severe tier. One is p.144, discrepancies in the stock statements submitted to the bank: the lexicon missed it because the report spells it "discrepencies". We did not change the lexicon after seeing this, to keep the result honest.

**Caveats:**
- **Small sample.** 8 reports (4 distressed, 4 healthy). An AUC from 16 pairs has a wide error bar; read it as indicative.
- **The only wrongly ordered pair is Gensol vs Titan.** Titan's 3 severe findings are real: credit risk on its IL&FS exposure after rating downgrades.
- **Some severe matches are noise:**
  - Yes Bank: "downgraded World GDP growth"; share acquisitions caught by the pledge-invocation term;
  - Maruti: a routine going-concern statement;
  - NTPC: winding-up of a small joint venture.
- **Bank reports use "fraud" and "NPA" routinely.** A sector-aware lexicon is future work.
- **The validation companies had been seen before.** RCOM, Titan and NTPC chose the classifier threshold, but not the lexicon.

**Takeaway.** RedFlag Lens is a screening tool. It finds the sentences an analyst should read and lists the most serious ones first. The severe-sign rate is a useful first signal at report level, but 8 reports cannot validate it as a distress predictor.

### 8.1 Fresh test: two reports never used anywhere (25 Nov 2026)

To check the pipeline on documents outside every split, we downloaded two new reports (`data/raw/new_test/`, git-ignored) and ran the app model (FinBERT, no LLM) with `src/eval/new_reports.py`. Output: `reports/backtest/new_unseen/`.

| Report | Group | Pages | Sentences | Flags / 1k | Severe | Severe / 1k | Time |
|---|---|---|---|---|---|---|---|
| SpiceJet FY2023 | distressed (airline, negative net worth) | 217 | 2,515 | 128.4 | 34 | **13.52** | 142 s |
| Nestlé India FY2024 | healthy | 274 | 3,080 | 31.5 | 5 | **1.62** | 123 s |

- **SpiceJet** scores above every report in the 8-report backtest. The severe findings include the auditor's "Material Uncertainty Related to Going Concern" (p.76, p.94, p.148), the qualified opinion (p.80, p.152), "unable to pay Rs. 750.00 million" (p.25), a stay application against the bank declaring the loan NPA (p.118), lessors' Section 9 IBC petitions (p.131) and negative net worth (p.145). About 6 of the 34 are false alarms: the cargo business transfer "as a going concern" (p.25, p.33, p.37, p.132, p.200) and the standard IBC disclosure heading (p.30).
- **Nestlé** falls in the healthy range. All 5 severe findings are false alarms: an NCLT merger scheme (p.108, p.146), the standard revenue-fraud key audit matter (p.114), a business transfer agreement (p.134) and CARO clause (xix) (p.122).
- **Lesson:** the severe lexicon should distinguish NCLT *schemes* and business transfers "as a going concern" from insolvency and going-concern *doubt*. We did not change the lexicon after this test.

**Re-run with the cascade (`REDFLAG_MODEL=cascade`, `reports/backtest/new_unseen/cascade/`).** Every severe finding was checked by hand.

| Report | Flags / 1k | Severe | Severe / 1k | Severe findings that are real | Time |
|---|---|---|---|---|---|
| SpiceJet FY2023 | 110.1 | 25 | **9.94** | **25 of 25**, covering 8 separate problems | 260 s |
| Nestlé India FY2024 | 16.2 | 2 | **0.65** | 0 of 2 (both the NCLT merger scheme) | 194 s |

- All nine weak or false SpiceJet severe findings from the FinBERT run disappeared (cargo business transfer "as a going concern" ×5, the IBC disclosure heading, "will continue as a going concern" ×2, the one-time-gain sentence, a ratio-table line), and all 8 real problems are still found: going-concern material uncertainty, "unable to pay Rs. 750.00 million", the qualified opinion, net loss and negative net worth, lessors' IBC petitions, the bank's move to classify the loan as NPA, Rs. 15,549 million of doubtful recoverables, and material weakness in internal controls.
- Nestlé's false alarms fell from 5 to 2; the remaining two are the NCLT merger scheme, a lexicon problem rather than a classifier problem.
- Precision of the severe tier across both reports rose from about 70% (FinBERT) to 25 of 27 (93%).
