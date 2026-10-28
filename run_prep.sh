#!/usr/bin/env bash
# Rebuild the whole dataset from the PDFs in data/raw/inbox (about 4 minutes on a laptop).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY -W ignore src/prep/01_ingest.py
$PY -W ignore src/prep/02_extract.py
$PY -W ignore src/prep/03_sentences.py
# 04 draws the pilot sample; run it only once so pilot labels stay valid:
[ -f data/annotations/pilot_300_labelled.jsonl ] || $PY -W ignore src/prep/04_split_sample.py
$PY -W ignore src/prep/05_eda.py > /dev/null
echo "done: data/interim/sentences.jsonl, reports/figures/, reports/eda_stats.json"
