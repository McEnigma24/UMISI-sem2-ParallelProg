#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python _src/compare.py

cd report
xelatex -interaction=nonstopmode report.tex >/dev/null
xelatex -interaction=nonstopmode report.tex
echo "PDF: $(realpath report.pdf)"