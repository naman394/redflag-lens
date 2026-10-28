#!/usr/bin/env bash
# Download the annual-report PDFs listed in data/raw/candidates.csv from the BSE public archive.
# Polite: one request per second. PDFs are not stored in git.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/raw/inbox
tail -n +2 data/raw/candidates.csv | while IFS=, read -r file scrip mmyy group; do
  out="data/raw/inbox/${file}.pdf"
  [ -s "$out" ] && continue
  url="https://www.bseindia.com/bseplus/AnnualReport/${scrip}/${scrip}${mmyy}.pdf"
  if curl -sfL --max-time 240 -o "$out" "$url" && file "$out" | grep -q PDF; then echo "ok   $file"; else rm -f "$out"; echo "miss $file"; fi
  sleep 1
done
# Gensol FY2024 (filed as a BSE corporate announcement)
[ -s data/raw/inbox/GENSOL_FY2024.pdf ] || curl -sfL -o data/raw/inbox/GENSOL_FY2024.pdf "https://bseindia.com/xml-data/corpfiling/AttachHis/cf949692-2280-4ee7-ba6f-30d1961f1ad5.pdf"
