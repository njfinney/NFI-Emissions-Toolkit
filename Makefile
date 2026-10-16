# NFI pipeline: Comext extract -> classify -> checks + review page.
# Run `make` after changing vocabulary.csv / overrides.csv, `make all` after new Comext files.
#   make DAT="path/to/full_*2026*.dat" PERIOD="Jun-Aug 2026" FX=0.8629 all

PY     ?= python3
DAT    ?= comext_dat/full_*.dat
PERIOD ?= Jun-Aug 2026
FX     ?= 0.8629
META    = https://ec.europa.eu/eurostat/api/dissemination/files/?file=comext%2FCOMEXT_METADATA%2FCLASSIFICATIONS_AND_RELATIONS

.PHONY: run all weights classify review test metadata

run: classify review test          # vocabulary or override change: no Comext rerun needed

all: weights run                   # new Comext months, new FX or new CN year

weights:
	$(PY) comext_unit_weights.py "$(DAT)" --codes codes_nfi.csv --out unit_weights.csv

classify:
	$(PY) classify.py NFI_Categorisations.csv --comext unit_weights.csv --usd_to_eur $(FX)

review:
	$(PY) build_review.py --comext unit_weights.csv --usd_to_eur $(FX) --comext_period "$(PERIOD)"

test:
	$(PY) -m pytest -q tests

metadata:                          # each January, when the new CN year starts
	mkdir -p comext_metadata
	curl -sSf -o comext_metadata/CN8-SUPPL_UNIT.txt "$(META)%2FRELATIONS%2FCN8-SUPPL_UNIT.txt"
	curl -sSf -o comext_metadata/SUPPL_UNIT.txt "$(META)%2FCLASSIFICATIONS%2FENGLISH%2FSUPPL_UNIT.txt"
	curl -sSf -o comext_metadata/CN.txt "$(META)%2FCLASSIFICATIONS%2FENGLISH%2FCN.txt"
