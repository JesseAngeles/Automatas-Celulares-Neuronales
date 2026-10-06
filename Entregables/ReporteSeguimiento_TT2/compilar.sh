#!/usr/bin/env bash
# Compila los reportes de seguimiento mensuales
set -e
cd "$(dirname "$0")"
for tex in [0-9]*ReporteSeguimiento.tex; do
	for _ in 1 2; do
		pdflatex -interaction=nonstopmode -halt-on-error "$tex" > /dev/null
	done
done
rm -f ./*.aux ./*.log ./*.out
