# RedFlag Lens

Reads an Indian company's annual report and lists its red flags for **credit analysts at banks and NBFCs**: going-concern doubts, auditor qualifications, defaults, insolvency proceedings, regulator penalties and disputed dues. Every finding links to its page, with the sentence highlighted. NLP course project, Industry Track.

## Results (test set: 5 companies never seen in training)

| Model | Precision | Recall | F1 | Speed |
|---|---|---|---|---|
| Keyword rules (Ctrl+F) | 0.10 | 0.63 | 0.17 | instant |
| TF-IDF + logistic regression | 0.44 | 0.69 | 0.54 | 0.2 ms/sentence |
| Zero-shot Qwen2.5-7B (no training) | 0.37 | 0.63 | 0.47 | 126 ms/sentence (GPU) |
| **FinBERT fine-tuned** | **0.64** | **0.875** | **0.74** | 45 ms/sentence (laptop CPU) |

FinBERT over 3 seeds: F1 0.71 ± 0.04, recall 0.85 ± 0.04. A full report takes about 2.5 minutes on a laptop CPU.

**Best model, on the 600-sentence test set v1.1** (37 risk sentences, 5 unseen companies):
- DeBERTa-v3-large with extra non-risk training examples, 3-seed ensemble, chosen on validation.
- F1 **0.86** (0.76–0.93), recall **0.92**, PR-AUC 0.89.
- The FinBERT in the app scores F1 0.77 and is about 7.6× faster on a laptop CPU (~2.5 min against ~16 min per report).
- The saved single DeBERTa model (`models/debv3l_aug`, F1 0.84, recall 0.89) runs with `REDFLAG_MODEL=deberta`; it is best on a GPU.
- See TECHNICAL_REPORT §4.6–4.7.

**Report-level backtest** (8 full reports, 4 distressed and 4 healthy, `reports/backtest/`):
- **Severe warning signs per 1,000 sentences** (going concern, default, audit qualification, insolvency, fraud and similar) separate distressed from healthy reports with **ROC-AUC 0.94**.
- The raw flag count scores only 0.50.
- On **Gensol FY2024**, held out from everything, the summary leads with the auditor's adverse remark on a subsidiary and a going-concern doubt.

See docs/TECHNICAL_REPORT.md §8 for caveats.

Details, error analysis, robustness and limitations: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md). Every run is logged in [reports/experiments.csv](reports/experiments.csv).

## Run the demo

```bash
/opt/homebrew/bin/python3.13 -m venv .venv          # Python 3.13 (PyTorch has no 3.15 build yet)
.venv/bin/pip install -r requirements.txt
src/prep/00_download.sh                             # fetch the 26 annual reports from BSE (needed for the samples)

.venv/bin/streamlit run src/app/ui.py               # dashboard → http://localhost:8501
.venv/bin/uvicorn src.app.api:app --port 8000       # API → POST /analyze, POST /feedback, GET /health
```

In the dashboard, pick **GENSOL_FY2024** and press **Analyse**. Gensol was held out from all training and testing; the company collapsed in 2025.

- **Optional:** put `GEMINI_API_KEY=...` in `.env` for an LLM summary in which every claim is checked against the sentences it cites. Without it, the summary is built directly from the findings.
- **Docker:** `docker build -t redflaglens . && docker run -p 8501:8501 -p 8000:8000 redflaglens`
- **API example:** `curl -F file=@data/raw/inbox/GENSOL_FY2024.pdf -F company=GENSOL localhost:8000/analyze`

## Reproduce

```bash
./run_prep.sh                                        # PDFs → 96,792 sentences, fragments flagged (~4 min)
.venv/bin/python src/label/build_sets.py             # sample labelling sets
.venv/bin/python src/label/llm_label.py data/labels/trainval_new.jsonl --model gemini-4-neon   # needs GEMINI_API_KEY
.venv/bin/python src/label/consensus.py              # freeze dataset v1.0 (data/labels/v1.0/)
.venv/bin/python src/models/baselines.py             # keyword + TF-IDF
.venv/bin/python src/models/finbert.py --seed 42     # GPU recommended (~25 s on an RTX 4090)
# Model weights are not in git (too large for GitHub). models/finbert (210 MB) comes from the line above;
# models/debv3l_aug (849 MB) from:
#   python src/models/encoder.py --base microsoft/deberta-v3-large --name debv3l --lr 1e-5 --epochs 5 --seed 42 --aug --save
.venv/bin/python src/eval/evaluate.py                # comparison, errors, robustness, latency → reports/eval/
.venv/bin/python src/models/build_boilerplate.py     # boilerplate list from train companies → models/boilerplate.txt
.venv/bin/python src/models/augment.py               # extra non-risk training examples (train companies only)
.venv/bin/python src/models/encoder.py --base microsoft/deberta-v3-large --name debv3l --lr 1e-5 --epochs 5 --seed 42   # GPU
.venv/bin/python src/models/ensemble.py              # compare configs + seed ensembles on validation
.venv/bin/python src/eval/backtest.py                # full pipeline on held-out reports → reports/backtest/
```

## Layout

```
src/prep/      PDF → text blocks → sentences → fragment flags; split; EDA; agreement checks
src/label/     labelling sets, LLM annotators, consensus → data/labels/v1.0/
src/models/    keyword, TF-IDF, FinBERT, zero-shot LLM, nil-statement rule, experiment logging
src/eval/      evaluation (M4) and report-level backtest
src/app/       pipeline, FastAPI service, Streamlit dashboard
models/        trained TF-IDF and FinBERT (fp16)
data/          manifest, splits, annotation guide v1.2, labels, data card
reports/       experiments.csv, eval/, backtest/, figures/
logs/          reports.jsonl (monitoring), feedback.jsonl (analyst votes)
docs/          technical report, research brief, product spec, milestone plan
```

## Data and labels

- **Corpus:** 26 annual reports (10 distressed, 16 healthy) from the BSE archive. The PDFs are not committed; `data/raw/manifest.csv` lists their URLs and checksums.
- **Labels (v1.0):**
  - Train/val: 1,850 sentences labelled by LLMs (Claude pilot + Gemini) using [the guide](data/annotations/guide.md).
  - Test: 300 sentences labelled independently by Claude and Gemini (κ = 0.90), with a third model breaking ties.
  - **These are not human gold labels**; see the [data card](data/DATA_CARD.md).
