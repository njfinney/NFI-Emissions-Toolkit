# NFI standardisation and unit-weight toolkit

## Files
| File | What it is |
|---|---|
| `classify.py` | Applies the rules, extracts sizes, estimates kg/unit, checks quantities |
| `vocabulary.csv` | The controlled item list: standard English name, CN code, weight method, multilingual match patterns. **Edit this to change behaviour.** |
| `overrides.csv` | Exact description → item_id (and optional kg) for one-offs and corrections. Applied before the rules. |
| `comext_unit_weights.py` | Reads Eurostat Comext `.dat` files → median kg per supplementary unit, the unit itself, and EUR per kg by CN8 |
| `comext_metadata/` | Eurostat `CN8-SUPPL_UNIT.txt`, `SUPPL_UNIT.txt`, `CN.txt` (bulk download, `comext/COMEXT_METADATA/CLASSIFICATIONS_AND_RELATIONS/`) |
| `codes_nfi.csv` | The CN codes used in the vocabulary, ready for the Comext script |
| `build_ceda_long.py` | Reshapes the Open CEDA workbook into long tables (`ceda_*.csv`) |
| `ceda_ef_long.csv`, `ceda_regional_ef_long.csv`, `ceda_sectors.csv`, `ceda_countries.csv` | Open CEDA 2025 in long format (also as `OpenCEDA_2025_long.xlsx`) |
| `country_iso3.csv` | Supplier country name → ISO3 → UN subregion, and whether CEDA covers it |

## Run
```
pip install pandas duckdb
python comext_unit_weights.py "full_v2_2025*.dat" --codes codes_nfi.csv --out unit_weights.csv
python build_ceda_long.py Open_CEDA_2025_updated_2025-11-12.xlsx   # only when CEDA changes
python classify.py NFI_Categorisations.csv --comext unit_weights.csv --usd_to_eur 0.8629
```
Without `--comext` the classifier still runs; Comext-method items just stay blank.
`--usd_to_eur` is EUR per USD for the Comext period (0.8629 = 1 / 1.15888, the mean of ECB daily
USD/EUR reference rates 2 Jan - 15 Oct 2026; the 2026 annual average is not yet published).

### Comext unit matching (`comext_route` and `kg_basis`)
The NFI Quantity is taken to be pieces. Only CN8 codes under the item's CN code that carry the
dominant supplementary unit are used.
| Supplementary unit | Route | kg per NFI unit |
|---|---|---|
| p/st (`PST`) | `COMEXT_PST` | Comext median kg/unit |
| pa (`PA`) | `COMEXT_PA` | kg per pair; footwear/gloves rows not described as pairs get `PAIR_QTY_CHECK` |
| 100 / 1000 p/st | `COMEXT_100_PST` / `COMEXT_1000_PST` | median / 100 or / 1000 |
| m, m2, m3, l | `COMEXT_CONVERTED` | median x `length_m`, `area_m2`, `volume_l`/1000 or `volume_l` parsed from the description; not applied without a size |
| none (kg only), or a unit with no size (incl. `CEEL`) | `PRICE_PROXY` | unit price x usd_to_eur / Comext EUR per kg |

Special cases:
- **Tiles and drywall** (`QTY_IN_M2_ITEMS`): the quantity is recorded in m2, so the Comext kg per m2
  is the kg per unit (`COMEXT_CONVERTED`); `QTY_NOT_INTEGER` is not raised for them.
- **Generators**: CN8 band from the rated output (kVA; kW / 0.8; watts). Diesel 8502.11-13 by
  default, spark ignition 8502.20 when the text says petrol / inverter / portable. With no rating
  the price proxy is used, since kg per set varies 100x across bands.
- **Price proxy on lump sums**: total kg = value x FX / EUR per kg; when units are re-estimated
  the total is spread over them, never multiplied again.

`COMEXT_PRICE_MISMATCH`: WFP price per kg (after FX) is outside 0.2-5x Comext EUR per kg for the same
CN8 codes (`comext_price_ratio`). It usually means the wrong CN8 within the 6-digit code or a
quantity-unit mismatch. Not computed for the price proxy (ratio 1 by construction).

## How it works
1. **No language detection, no free translation.** Each item in `vocabulary.csv` carries
   EN/FR/ES/PT/typo variants as a regex. Text is accent-stripped and lower-cased first.
2. **Head-noun matching.** Text after "for / pour / para / with / avec / con" is ignored for
   most rules, so "Heater, wood burning stove, for tent" is a heater, not a tent.
3. **First match wins, in file order.** Services and kits sit above components, specific
   items above generic ones. Every row records the rule and the words that matched.
4. **Standard name = fixed item name + at most one size** (e.g. `Tarpaulin, 4x6 m`,
   `Family tent, 23 m2`, `Cooking pot, 110 L`). Size never enters the CN mapping.
5. **Weight hierarchy** (best first): weight stated in text → humanitarian catalogue
   (IFRC/UNHCR) → parametric (area × g/m², area × thickness × density, rope length × linear
   mass) → Comext median kg per supplementary unit → **price proxy** (unit price ÷ Comext EUR/kg
   for the CN code) → blank.
6. **Quantity sanity checks.** Price per kg (or per unit) is compared with the item's own
   median. Flags: `LUMP_SUM` (qty 1, large value), `QTY_UNIT_SUSPECT` (e.g. hygiene kits
   bought "per pallet"), `QTY_NOT_INTEGER` (probably m² or kg). Units are only re-estimated
   for lump sums and palletised kits; everything else is flagged, not changed.

## Output columns worth knowing
`item_std` standard name · `scope` GOODS / KIT / WORKS / SERVICE / ADJUSTMENT / OUT_OF_SCOPE / UNCLEAR ·
`emissions_approach` · `cn` · `ceda_target` · `kg_per_unit` · `kg_basis` (where the number came from) ·
`confidence_score` 0-100 · `confidence_band` · `confidence_basis` (the three factors) ·
`flags` · `units_est` · `total_kg_est` · `match` + `matched_text` (why the row got its item).

## Maintaining it
Run everything with `make` from this folder (`make run` after a vocabulary or override change,
`make all DAT=... PERIOD=... FX=...` after new Comext months). Each run writes `nfi_checks.csv` and
`nfi_review.html`, and stops if an integrity check fails.

| When | Do |
|---|---|
| A description lands in the wrong item | Add a pattern to the right row of `vocabulary.csv` (one-off: a line in `overrides.csv`), then add the description to `GOLDEN` in `tests/test_pipeline.py` |
| New item type | Add a row above anything more generic, add its CN to `codes_nfi.csv`, `make all` |
| New NFI extract | Replace `NFI_Categorisations.csv`, `make run`, work through the review page's Checks tab |
| New Comext months | Put the `.dat` files in `comext_dat/` (not committed), `make all PERIOD=...` |
| Each run | Set `FX` to the ECB average for the Comext period |
| Each January | `make metadata`, then `make test`: the CN-year test lists vocabulary codes that were retired |

Review each change as a pull request: the diff of `nfi_lookup.csv` shows exactly which
descriptions changed item, weight or score.

### Review page
`nfi_review.html` is self-contained (open it in a browser, or publish it). Tabs: Overview
(value by approach, confidence bands, WFP vs Comext price per kg, where the weight sits), Checks
(integrity rules that must be 0, review flags, info), Rows (filter, sort, row detail with the
score breakdown), Items (spread of kg/unit and price per kg inside each item).

## Emissions approach (first rule that applies)
1. **Unallocated** - scope UNCLEAR, contract adjustments (contingency, variation, retention, tax,
   overhead), or zero / negative value.
2. **Spend-based (OpenCEDA)** - services, works, and out-of-scope purchases (food, medicines).
3. **Average Data (Research Derived)** - goods/kits with a kg/unit from the description, a
   humanitarian catalogue, a parametric calculation, an engineering standard, a reviewer, or an
   Unqualified Estimate.
4. **Average Data (COMEXT)** - goods with a CN code but no research-derived weight (Comext
   median kg/unit, or the price proxy).
5. **Spend-based (OpenCEDA)** - goods with neither a weight nor a CN code (fallback).

## Confidence score (0-100)
`confidence_score = 100 x identification x quantification x quantity`

| Factor | Evidence | Score |
|---|---|---|
| Identification | Reviewed override | 1.00 |
| | Head-noun rule match | 0.90 |
| | Keyword match anywhere in text | 0.85 |
| | Fallback match (head-noun pass failed) | 0.60 |
| | Inferred from material group only | 0.40 |
| | Catch-all item (Tent unspecified, Other kit, ...) | capped at 0.60 |
| | Unclear | 0 |
| Quantification (goods: weight) | Weight stated in description | 1.00 |
| | Catalogue, same specification | 0.90 |
| | Parametric, stated dimensions + material property | 0.85 |
| | Reviewer-entered weight | 0.80 |
| | Parametric, stated size x published rate | 0.75 |
| | Catalogue, type assumed / engineering standard / quantity-in-kg inference | 0.70 |
| | Comext median kg/unit (0.40 if p75/p25 > 3) | 0.60 |
| | Comext kg per m / m2 / m3 / L x size from description (0.40 if p75/p25 > 3) | 0.50 |
| | Unqualified Estimate / Comext price proxy | 0.30 |
| | No weight yet | 0 |
| Quantification (services: sector fit) | Sector describes the service exactly | 0.90 |
| | Right sector family, detail assumed (e.g. road mode) | 0.70-0.80 |
| | Several sectors plausible | 0.50-0.60 |
| | Unclear service | 0.30 |
| Geography (services only, multiplies sector fit) | Supplier-country factor | 1.00 |
| | UN-subregion average (country not in CEDA) | 0.85 |
| | Rest of World | 0.70 |
| Quantity | As recorded, no flag; or not needed (spend-based, price proxy) | 1.00 |
| | Stated in text | 0.90 |
| | Flagged but unchanged | 0.80 |
| | Re-estimated from value / median unit price | 0.60 |

Bands: High >= 70 · Medium 40-69 · Low 1-39 · None = 0 (not yet quantified, or Unallocated).
The score measures how sure we are about this row's mapping and quantities; it does not rank
the calculation methods against each other.

## Open CEDA factors
- Every item has a `ceda_code` / `ceda_target`. For Spend-based rows it is the factor used; for goods it is
  only a fallback / cross-check.
- Factor = kg CO2e per USD, **2025 purchaser price**, chosen by supplier country, then UN subregion average,
  then Rest of World (`ceda_ef_geography` says which). `spend_kgco2e = Goods_Receipt_Value_USD x factor`.
  Use it only where `emissions_approach` is Spend-based; elsewhere it is a cross-check.
- Flag `SUPPLIER_IN_TRADE_HUB`: spend-based goods bought from a re-export hub (UAE, Belgium, Netherlands,
  Malaysia, Panama, Singapore). The supplier country may not be where the goods were made.
- In CEDA's own mapping, South Asia (India, Pakistan, Nepal, Bangladesh, Sri Lanka, Iran, Afghanistan, Bhutan)
  sits under "Central Asia", so that regional average mixes Central and South Asia.
- Attribution required: "CEDA by Watershed" (CC BY-SA 4.0).
