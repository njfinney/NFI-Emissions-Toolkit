#!/usr/bin/env python3
"""
Build the NFI review page and the checks table from the classifier outputs.

Usage:
  python build_review.py --prefix nfi --comext unit_weights.csv --usd_to_eur 0.8629 \
      --comext_period "Jun-Aug 2026"
Outputs: nfi_checks.csv (one row per check, with counts), nfi_review.html (self-contained page)
Exit code 1 if any integrity check fails, so a pipeline run stops on a broken output.
"""
import argparse
import datetime as dt
import json
import math

import pandas as pd

# ---------------------------------------------------------------- checks
# Each check is (id, severity, title, why it matters, row mask function).
# integrity = the outputs contradict the toolkit's own rules (must be 0)
# review    = plausible data or mapping problem, look at the rows
# info      = context, no action needed by default

RESEARCH, COMEXT = "Average Data (Research Derived)", "Average Data (COMEXT)"
ROUTE_SCORES = {"COMEXT_PST": {0.6, 0.4}, "COMEXT_PA": {0.6, 0.4}, "COMEXT_100_PST": {0.6, 0.4},
                "COMEXT_1000_PST": {0.6, 0.4}, "COMEXT_CONVERTED": {0.5, 0.4}, "PRICE_PROXY": {0.3}}


def flag(o, name):
    return o["flags"].fillna("").str.split("|").apply(lambda f: name in f)


def build_checks(o, eur_per_kg, fx):
    kb = o["kg_basis"].fillna("").astype(str)
    goods = o["scope"].isin(["GOODS", "KIT"])
    proxy = kb.str.startswith("price proxy")
    tot_kg = o["total_kg_est"].sum()
    proxy_total = o["Goods_Receipt_Value_USD"] * fx / o["cn"].map(eur_per_kg)
    usd_per_kg = o["unit_price_usd"] / o["kg_per_unit"]
    spread = o["comext_p75"] / o["comext_p25"]
    return [
        ("SCORE_FORMULA", "integrity", "Score is not 100 x ID x weight x quantity",
         "confidence_score must equal the product of its three factors.",
         lambda: o.id_score.notna() & (o.confidence_score != (100 * o.id_score * o.quant_score * o.qty_factor).round())),
        ("ROUTE_SCORE", "integrity", "Weight score does not match the Comext route",
         "PST/PA 0.6 or 0.4, converted 0.5 or 0.4, price proxy 0.3.",
         lambda: (o.emissions_approach == COMEXT) & o.comext_route.notna() & o.quant_score.notna() & pd.Series(
             [q not in ROUTE_SCORES.get(r, {q}) for r, q in zip(o.comext_route, o.quant_score)], index=o.index)),
        ("APPROACH_BASIS", "integrity", "Emissions approach contradicts the weight source",
         "COMEXT rows need a Comext or price-proxy weight; research-derived rows must not have one.",
         lambda: ((o.emissions_approach == COMEXT) & o.kg_per_unit.notna() & ~kb.str.startswith(("Comext", "price proxy")))
         | ((o.emissions_approach == RESEARCH) & kb.str.startswith(("Comext", "price proxy")))),
        ("TOTAL_KG", "integrity", "Total kg is not units x kg per unit",
         "total_kg_est must equal units_est x kg_per_unit (rounding allowed).",
         lambda: o.total_kg_est.notna() & ((o.total_kg_est - o.units_est * o.kg_per_unit).abs() > 1 + 0.002 * o.total_kg_est.abs())),
        ("PROXY_TOTAL", "integrity", "Price-proxy total kg is not value x FX / EUR per kg",
         "For the price proxy the quantity cancels out; a difference means the order value was counted twice.",
         lambda: proxy & proxy_total.notna() & ((o.total_kg_est - proxy_total).abs() > 1 + 0.01 * proxy_total)),
        ("COMEXT_PRICE_MISMATCH", "review", "WFP price per kg is 0.2-5x off Comext",
         "Usually the wrong CN8 within the 6-digit code, or a quantity in another unit.",
         lambda: flag(o, "COMEXT_PRICE_MISMATCH")),
        ("QTY_UNIT_SUSPECT", "review", "Price per kg far from the item's own median",
         "The quantity may be in kg, m, m2 or pallets rather than pieces.", lambda: flag(o, "QTY_UNIT_SUSPECT")),
        ("LUMP_SUM", "review", "Quantity 1 with a large value",
         "Units were re-estimated from the item's typical unit price.", lambda: flag(o, "LUMP_SUM")),
        ("KG_CONCENTRATION", "review", "One row carries over 2% of all estimated kg",
         "Large single contributions are where magnitude errors hide.",
         lambda: o.total_kg_est > 0.02 * tot_kg),
        ("PRICE_PER_KG_EXTREME", "review", "Implied price below $0.05/kg or above $2,000/kg",
         "Physically unlikely for goods; check the quantity and the weight.",
         lambda: goods & usd_per_kg.notna() & ((usd_per_kg < 0.05) | (usd_per_kg > 2000))),
        ("GOODS_NO_WEIGHT", "review", "Goods or kit without a kg per unit",
         "These fall back to spend-based factors.", lambda: goods & o.kg_per_unit.isna() & (o.emissions_approach != "Unallocated")),
        ("WEAK_MATCH", "review", "Item inferred from a fallback match or the material group",
         "The description did not match on its head noun; check the item.",
         lambda: o["match"].isin(["rule:full-fallback", "material-group-prior"])),
        ("PAIR_QTY_CHECK", "review", "Weight is per pair but the text does not say pairs",
         "Footwear and gloves: the quantity may count single items.", lambda: flag(o, "PAIR_QTY_CHECK")),
        ("QTY_NOT_INTEGER", "review", "Quantity is not a whole number",
         "Usually m2, m or kg rather than pieces.", lambda: flag(o, "QTY_NOT_INTEGER")),
        ("WIDE_COMEXT", "info", "Comext spread is wide (p75/p25 > 3)",
         "The median is used but the item type varies a lot within the code.",
         lambda: o.kg_basis.fillna("").str.startswith("Comext") & (spread > 3)),
        ("GENERIC_ITEM", "info", "Catch-all item (tent unspecified, other kit)",
         "Identification is capped at 0.6.", lambda: flag(o, "GENERIC_ITEM")),
        ("UNQUALIFIED", "info", "Weight is an Unqualified Estimate",
         "No published source for this weight.", lambda: kb.str.contains("Unqualified Estimate")),
        ("TRADE_HUB", "info", "Spend-based goods bought from a re-export hub",
         "The supplier country may not be where the goods were made.", lambda: flag(o, "SUPPLIER_IN_TRADE_HUB")),
    ]


def clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, float):
        return round(v, 4)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="nfi")
    ap.add_argument("--comext", default="unit_weights.csv")
    ap.add_argument("--usd_to_eur", type=float, default=0.8629)
    ap.add_argument("--comext_period", default="Jun-Aug 2026")
    ap.add_argument("--out", default=None, help="default: <prefix>_review.html")
    args = ap.parse_args()

    o = pd.read_csv(f"{args.prefix}_classified.csv", dtype={"cn": str})
    cx = pd.read_csv(args.comext, dtype={"cn8": str})
    eur_per_kg = {}
    for c in o.cn.dropna().unique():
        sub = cx[cx.cn8.str.startswith(str(c))].eur_per_kg.dropna()
        eur_per_kg[c] = float(sub.median()) if len(sub) else None

    checks, row_checks = [], [[] for _ in range(len(o))]
    w = o.Goods_Receipt_Value_USD
    for cid, sev, title, why, fn in build_checks(o, eur_per_kg, args.usd_to_eur):
        m = pd.Series(fn(), index=o.index).fillna(False).astype(bool)
        for i in o.index[m]:
            row_checks[i].append(cid)
        checks.append(dict(id=cid, severity=sev, title=title, why=why, rows=int(m.sum()),
                           value_usd=float(w[m].sum()), value_share=float(w[m].sum() / w.sum())))
    pd.DataFrame(checks).to_csv(f"{args.prefix}_checks.csv", index=False)

    cols = {"desc": "Detailed_Description", "item_std": "item_std", "item_id": "item_id", "group": "group",
            "scope": "scope", "approach": "emissions_approach", "cn": "cn", "supp_unit": "supp_unit",
            "route": "comext_route", "kg": "kg_per_unit", "kg_basis": "kg_basis", "qty": "Quantity",
            "units": "units_est", "units_basis": "units_basis", "total_kg": "total_kg_est",
            "value": "Goods_Receipt_Value_USD", "unit_price": "unit_price_usd", "score": "confidence_score",
            "id_s": "id_score", "q_s": "quant_score", "qty_f": "qty_factor", "conf_basis": "confidence_basis",
            "flags": "flags", "match": "match", "matched": "matched_text", "attributes": "attributes",
            "country": "Supplier_Country_Name", "mg": "Material_Group_Description", "p25": "comext_p25",
            "p75": "comext_p75", "cx_eur_kg": "comext_eur_per_kg", "ratio": "comext_price_ratio",
            "ceda_ef": "ceda_ef_kgco2e_per_usd", "ceda_geo": "ceda_ef_geography", "spend_co2": "spend_kgco2e"}
    keys = list(cols)
    rows = [[clean(r[c]) for c in cols.values()] + [row_checks[i]] for i, r in o.iterrows()]
    meta = dict(generated=dt.datetime.now().strftime("%d %b %Y %H:%M"), comext_period=args.comext_period,
                usd_to_eur=args.usd_to_eur, rows=len(o), source=f"{args.prefix}_classified.csv")
    data = json.dumps(dict(keys=keys + ["checks"], rows=rows, checks=checks, meta=meta),
                      ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("/*__DATA__*/null", data)
    out = args.out or f"{args.prefix}_review.html"
    open(out, "w", encoding="utf-8").write(html)

    bad = [c for c in checks if c["severity"] == "integrity" and c["rows"]]
    print(f"Wrote {out} and {args.prefix}_checks.csv")
    for c in checks:
        print(f"  {c['severity']:9} {c['rows']:5} rows {c['value_share']:6.1%} value  {c['id']}")
    if bad:
        raise SystemExit(f"{len(bad)} integrity check(s) failed: " + ", ".join(c["id"] for c in bad))


TEMPLATE = r"""<title>NFI Weight Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a working review tool. Header band with run facts, four tabs, summary before detail. */
:root {
  --bg: #f4f5f2; --surface: #fcfcfb; --raise: #ffffff; --ink: #141513; --ink-2: #4d4f4a; --muted: #6b6d67;
  --line: #e0e1da; --axis: #c3c2b7; --accent: #1c5cab; --accent-wash: #e6eefa;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --crit: #d03b3b; --warn: #b77a00; --warn-dot: #fab219; --good: #0a8a0a; --info: #52514e;
  --crit-wash: #fbeaea; --warn-wash: #fdf4e0; --info-wash: #efefec; --good-wash: #e7f5e7;
  --font: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111211; --surface: #1a1a19; --raise: #20211f; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #a09f97;
  --line: #2c2c2a; --axis: #45453f; --accent: #86b6ef; --accent-wash: #1d2a3b;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --crit: #e66767; --warn: #fab219; --warn-dot: #fab219; --good: #0ca30c; --info: #c3c2b7;
  --crit-wash: #3a1f1f; --warn-wash: #3a2f15; --info-wash: #262624; --good-wash: #183018; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111211; --surface: #1a1a19; --raise: #20211f; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #a09f97;
  --line: #2c2c2a; --axis: #45453f; --accent: #86b6ef; --accent-wash: #1d2a3b;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --crit: #e66767; --warn: #fab219; --warn-dot: #fab219; --good: #0ca30c; --info: #c3c2b7;
  --crit-wash: #3a1f1f; --warn-wash: #3a2f15; --info-wash: #262624; --good-wash: #183018; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 var(--font); }
.wrap { max-width: 1240px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header { background: var(--surface); border-bottom: 1px solid var(--line); }
header .wrap { padding-block: 20px 0; display: grid; gap: 14px; }
h1 { font-size: 22px; line-height: 1.2; margin: 0; font-weight: 700; letter-spacing: -0.01em; text-wrap: balance }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--ink-2); font-size: 13px }
.meta b { color: var(--ink); font-weight: 600 }
nav { display: flex; gap: 4px; overflow-x: auto }
nav a { padding: 10px 14px; color: var(--ink-2); text-decoration: none; font-weight: 600; border-bottom: 2px solid transparent; white-space: nowrap }
nav a:hover { color: var(--ink) }
nav a[aria-current="page"] { color: var(--accent); border-bottom-color: var(--accent) }
nav a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, tr:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }
main .wrap { padding-block: 22px 48px; display: grid; gap: 22px }
section[hidden] { display: none }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 16px; min-width: 0 }
.panel h2 { font-size: 15px; margin: 0 0 4px; font-weight: 700 }
.panel .sub { color: var(--muted); font-size: 12.5px; margin: 0 0 12px }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px }
.kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; display: grid; gap: 2px }
.kpi .l { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600 }
.kpi .v { font-size: 24px; font-weight: 700; line-height: 1.2 }
.kpi .n { font-size: 12.5px; color: var(--ink-2) }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 16px }
svg text { fill: var(--muted); font: 11px var(--font) }
svg .ink { fill: var(--ink) } svg .ink2 { fill: var(--ink-2) }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--ink-2); margin-top: 10px }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px }
.legend b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums }
.tip { position: fixed; z-index: 20; pointer-events: none; background: var(--raise); color: var(--ink); border: 1px solid var(--line);
  box-shadow: 0 4px 16px rgba(0,0,0,.12); border-radius: 6px; padding: 8px 10px; font-size: 12.5px; max-width: 320px }
.tip .t { font-weight: 600; margin-bottom: 2px }
.tip .r { color: var(--ink-2); font-variant-numeric: tabular-nums }
.pill { display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; font-weight: 600; padding: 1px 8px; border-radius: 999px; white-space: nowrap }
.pill::before { content: ""; width: 7px; height: 7px; border-radius: 50% }
.pill.integrity { background: var(--crit-wash); color: var(--crit) } .pill.integrity::before { background: var(--crit); border-radius: 1px }
.pill.review { background: var(--warn-wash); color: var(--warn) } .pill.review::before { background: var(--warn-dot); clip-path: polygon(50% 0, 100% 100%, 0 100%); border-radius: 0 }
.pill.info { background: var(--info-wash); color: var(--info) } .pill.info::before { background: var(--info) }
.pill.pass { background: var(--good-wash); color: var(--good) } .pill.pass::before { background: var(--good) }
.checks { display: grid; gap: 8px }
.check { display: grid; grid-template-columns: 1fr auto; gap: 4px 16px; align-items: center; padding: 12px 14px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; cursor: pointer; text-align: left; font: inherit; color: inherit; width: 100% }
.check:hover { border-color: var(--axis) }
.check .h { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-weight: 600; min-width: 0 }
.check .why { grid-column: 1; color: var(--ink-2); font-size: 13px }
.check .num { grid-row: 1 / span 2; grid-column: 2; text-align: right; font-variant-numeric: tabular-nums }
.check .num b { display: block; font-size: 18px }
.check .num span { color: var(--muted); font-size: 12px }
.check.zero { cursor: default; opacity: .75 }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: end }
.filters label { display: grid; gap: 3px; font-size: 11.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .05em }
input, select { font: 14px var(--font); color: var(--ink); background: var(--raise); border: 1px solid var(--line); border-radius: 5px; padding: 6px 8px; min-width: 0 }
input[type=search] { width: min(320px, 100%) }
button.btn { font: 600 13px var(--font); color: var(--accent); background: var(--accent-wash); border: 0; border-radius: 5px; padding: 7px 12px; cursor: pointer }
.active-filter { display: flex; flex-wrap: wrap; gap: 6px; font-size: 12.5px; color: var(--ink-2) }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface) }
table { border-collapse: collapse; width: 100%; font-size: 13px }
th { text-align: left; font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); font-weight: 600;
  padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; cursor: pointer; user-select: none; background: var(--surface); position: sticky; top: 0 }
th[aria-sort="ascending"]::after { content: " ▲" } th[aria-sort="descending"]::after { content: " ▼" }
td { padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap }
td.code { font-family: var(--mono); font-size: 12px; white-space: nowrap }
td.desc { min-width: 220px; max-width: 360px }
tbody tr { cursor: pointer } tbody tr:hover { background: var(--accent-wash) }
tr.sel { background: var(--accent-wash) }
.tags { display: flex; flex-wrap: wrap; gap: 4px }
.tag { font-size: 11px; padding: 0 6px; border-radius: 3px; background: var(--info-wash); color: var(--ink-2); white-space: nowrap }
.tag.integrity { background: var(--crit-wash); color: var(--crit) } .tag.review { background: var(--warn-wash); color: var(--warn) }
.score { display: inline-block; min-width: 30px; text-align: right; font-weight: 600 }
.bar-s { display: inline-block; width: 44px; height: 6px; border-radius: 3px; background: var(--line); margin-left: 6px; vertical-align: 1px; overflow: hidden }
.bar-s i { display: block; height: 100%; background: var(--s1) }
.pager { display: flex; gap: 10px; align-items: center; justify-content: space-between; flex-wrap: wrap; font-size: 13px; color: var(--ink-2); padding-top: 10px }
.detail { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); gap: 4px 24px }
.detail dl { margin: 0; display: grid; grid-template-columns: 130px 1fr; gap: 4px 10px; font-size: 13px; align-content: start }
.detail dt { color: var(--muted) } .detail dd { margin: 0; overflow-wrap: anywhere }
.detail h3 { font-size: 16px; margin: 0 0 10px; grid-column: 1 / -1 }
.mono { font-family: var(--mono); font-size: 12px }
.empty { color: var(--muted); padding: 16px }
@media (prefers-reduced-motion: no-preference) { tbody tr, .check { transition: background .12s, border-color .12s } }
</style>

<header><div class="wrap">
  <h1>NFI Weight Review</h1>
  <div class="meta" id="meta"></div>
  <nav aria-label="Sections">
    <a href="#overview" data-tab="overview">Overview</a>
    <a href="#checks" data-tab="checks">Checks</a>
    <a href="#rows" data-tab="rows">Rows</a>
    <a href="#items" data-tab="items">Items</a>
  </nav>
</div></header>

<main><div class="wrap">
<section id="tab-overview">
  <div class="kpis" id="kpis"></div>
  <div class="grid2" style="margin-top:16px">
    <div class="panel"><h2>Value by emissions approach</h2><p class="sub">Share of Goods Receipt value (USD)</p><div id="ch-approach"></div><div class="legend" id="lg-approach"></div></div>
    <div class="panel"><h2>Value by confidence band</h2><p class="sub">High 70+ · Medium 40-69 · Low 1-39 · None 0</p><div id="ch-band"></div></div>
    <div class="panel"><h2>WFP price per kg against Comext</h2><p class="sub">COMEXT rows with a Comext kg/unit, EUR per kg, log scales. Outside the shaded 0.2-5x band is flagged. Click a point to open the row.</p><div id="ch-scatter"></div><div class="legend" id="lg-scatter"></div></div>
    <div class="panel"><h2>Where the estimated weight sits</h2><p class="sub">Total estimated tonnes by item, top 12. A bar that looks too long is the first place to look.</p><div id="ch-kg"></div></div>
  </div>
</section>

<section id="tab-checks" hidden>
  <div class="panel"><h2>Consistency checks</h2>
  <p class="sub">Integrity checks test the outputs against the toolkit's own rules and must be zero. Review checks point at rows to look at. Select a check to see its rows.</p>
  <div class="checks" id="checks"></div></div>
</section>

<section id="tab-rows" hidden>
  <div class="filters">
    <label for="f-q">Search<input type="search" id="f-q" placeholder="Description, item, CN code"></label>
    <label for="f-ap">Approach<select id="f-ap"></select></label>
    <label for="f-ck">Check<select id="f-ck"></select></label>
    <label for="f-it">Item<select id="f-it"></select></label>
    <label for="f-sc">Max score<input type="number" id="f-sc" min="0" max="100" step="5" style="width:90px"></label>
    <button class="btn" id="f-clear" type="button">Clear filters</button>
  </div>
  <div class="active-filter" id="f-summary"></div>
  <div class="panel" id="detail" hidden></div>
  <div class="scroll"><table id="t-rows"><thead></thead><tbody></tbody></table></div>
  <div class="pager"><span id="p-info"></span><span><button class="btn" id="p-prev" type="button">Previous</button> <button class="btn" id="p-next" type="button">Next</button></span></div>
</section>

<section id="tab-items" hidden>
  <div class="panel"><h2>Consistency within items</h2>
  <p class="sub">One row per item. A wide range of price per kg inside one item means some rows have the wrong quantity unit, the wrong weight, or belong to another item. Select an item to see its rows.</p>
  <div class="filters" style="margin-bottom:10px"><label for="i-q">Search<input type="search" id="i-q" placeholder="Item name"></label></div>
  <div class="scroll"><table id="t-items"><thead></thead><tbody></tbody></table></div></div>
</section>
</div></main>
<div class="tip" id="tip" hidden></div>

<script>
const D = /*__DATA__*/null;
const K = Object.fromEntries(D.keys.map((k, i) => [k, i]));
const R = D.rows.map((r, i) => { const o = { i }; D.keys.forEach((k, j) => o[k] = r[j]); return o; });
const CK = Object.fromEntries(D.checks.map(c => [c.id, c]));
const APPROACHES = ["Average Data (Research Derived)", "Average Data (COMEXT)", "Spend-based (OpenCEDA)", "Unallocated"];
const AP_COLOR = ["--s1", "--s2", "--s3", "--s4"];
const $ = s => document.querySelector(s);
const fmtUSD = v => v == null ? "–" : "$" + (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(2) + "M" : Math.abs(v) >= 1e4 ? Math.round(v / 1e3) + "k" : Math.round(v).toLocaleString());
const fmtN = (v, d = 1) => v == null ? "–" : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : (+v.toFixed(d)).toLocaleString(undefined, { maximumFractionDigits: d });
const fmtKg = v => v == null ? "–" : v >= 1000 ? fmtN(v / 1000, 1) + " t" : fmtN(v, v < 1 ? 3 : 1) + " kg";
const pct = v => (100 * v).toFixed(1) + "%";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const totalV = R.reduce((a, r) => a + (r.value || 0), 0);
const shortAp = a => a.replace("Average Data ", "").replace(/[()]/g, "");

// ---------------------------------------------------------------- header + KPIs
const m = D.meta;
$("#meta").innerHTML = `<span><b>${R.length.toLocaleString()}</b> purchase lines</span><span>Comext <b>${esc(m.comext_period)}</b></span>
  <span>USD to EUR <b>${m.usd_to_eur}</b></span><span>Built <b>${esc(m.generated)}</b> from <span class="mono">${esc(m.source)}</span></span>`;
const wConf = R.reduce((a, r) => a + (r.score || 0) * (r.value || 0), 0) / totalV;
const withKg = R.filter(r => r.kg != null).reduce((a, r) => a + (r.value || 0), 0) / totalV;
const totKg = R.reduce((a, r) => a + (r.total_kg || 0), 0);
const integ = D.checks.filter(c => c.severity === "integrity");
const integBad = integ.filter(c => c.rows > 0);
const review = D.checks.filter(c => c.severity === "review");
const reviewRows = new Set(R.filter(r => r.checks.some(c => CK[c].severity === "review")).map(r => r.i));
$("#kpis").innerHTML = [
  ["Goods receipt value", fmtUSD(totalV), `${R.length.toLocaleString()} lines`],
  ["Confidence, value-weighted", wConf.toFixed(1), "out of 100"],
  ["Value with a kg per unit", pct(withKg), "goods, kits and services"],
  ["Estimated weight", fmtN(totKg / 1000, 0) + " t", "sum of total_kg_est"],
  ["Integrity checks", integBad.length ? `${integBad.length} failing` : "All pass",
    `${integ.length} rules · <span class="pill ${integBad.length ? "integrity" : "pass"}">${integBad.length ? "Fix first" : "Pass"}</span>`],
  ["Rows to review", reviewRows.size.toLocaleString(), `${review.length} review checks`],
].map(([l, v, n]) => `<div class="kpi"><span class="l">${l}</span><span class="v">${v}</span><span class="n">${n}</span></div>`).join("");

// ---------------------------------------------------------------- tooltip
const tip = $("#tip");
function showTip(ev, html) { tip.innerHTML = html; tip.hidden = false;
  const x = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8), y = Math.min(ev.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px"; }
function hideTip() { tip.hidden = true; }

// ---------------------------------------------------------------- charts
function approachChart() {
  const vals = APPROACHES.map(a => R.filter(r => r.approach === a).reduce((s, r) => s + (r.value || 0), 0));
  const W = 600, H = 44, gap = 2; let x = 0;
  const segs = vals.map((v, i) => { const w = Math.max(0, v / totalV * (W - gap * 3)); const s = { x, w, i, v }; x += w + gap; return s; });
  $("#ch-approach").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Value share by emissions approach">
    ${segs.map(s => `<rect data-i="${s.i}" x="${s.x}" y="6" width="${s.w}" height="32" rx="${s.w > 8 ? 4 : 1}" fill="var(${AP_COLOR[s.i]})" style="cursor:pointer"></rect>`).join("")}</svg>`;
  $("#lg-approach").innerHTML = APPROACHES.map((a, i) => `<span><i style="background:var(${AP_COLOR[i]})"></i>${shortAp(a)} <b>${pct(vals[i] / totalV)}</b> · ${fmtUSD(vals[i])}</span>`).join("");
  $("#ch-approach").querySelectorAll("rect").forEach(el => {
    const i = +el.dataset.i;
    el.addEventListener("mousemove", ev => showTip(ev, `<div class="t">${APPROACHES[i]}</div><div class="r">${pct(vals[i] / totalV)} of value · ${fmtUSD(vals[i])}</div><div class="r">Click to list rows</div>`));
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("click", () => { hideTip(); setFilters({ ap: APPROACHES[i] }); go("rows"); });
  });
}

function bandChart() {
  const bands = ["High", "Medium", "Low", "None"];
  const band = s => s >= 70 ? "High" : s >= 40 ? "Medium" : s > 0 ? "Low" : "None";
  const v = bands.map(b => R.filter(r => band(r.score || 0) === b).reduce((s, r) => s + (r.value || 0), 0));
  const n = bands.map(b => R.filter(r => band(r.score || 0) === b).length);
  const W = 600, rowH = 34, L = 74, Rr = 120, H = bands.length * rowH + 4, max = Math.max(...v);
  $("#ch-band").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Value by confidence band">
    ${bands.map((b, i) => { const w = v[i] / max * (W - L - Rr); const y = i * rowH + 6;
      return `<g data-i="${i}" style="cursor:pointer"><rect x="0" y="${y - 4}" width="${W}" height="${rowH}" fill="transparent"></rect>
      <text x="0" y="${y + 15}" class="ink2">${b}</text>
      <rect x="${L}" y="${y + 2}" width="${Math.max(w, 1)}" height="18" rx="${w > 8 ? 4 : 1}" fill="var(--s1)"></rect>
      <text x="${L + w + 8}" y="${y + 15}" class="ink">${pct(v[i] / totalV)} · ${n[i].toLocaleString()} lines</text></g>`; }).join("")}</svg>`;
  $("#ch-band").querySelectorAll("g").forEach(g => {
    const i = +g.dataset.i;
    g.addEventListener("mousemove", ev => showTip(ev, `<div class="t">${bands[i]} confidence</div><div class="r">${fmtUSD(v[i])} · ${n[i]} lines</div><div class="r">Click to list rows</div>`));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => { hideTip(); setFilters({ band: bands[i] }); go("rows"); });
  });
}

function scatter() {
  const pts = R.filter(r => r.ratio != null && r.cx_eur_kg > 0 && r.kg > 0 && r.unit_price > 0)
    .map(r => ({ r, x: r.cx_eur_kg, y: r.unit_price * m.usd_to_eur / r.kg, bad: r.checks.includes("COMEXT_PRICE_MISMATCH") }));
  const W = 600, H = 380, L = 52, B = 40, T = 10, Rr = 30;
  const lx = pts.map(p => Math.log10(p.x)), ly = pts.map(p => Math.log10(p.y));
  const lo = Math.floor(Math.min(...lx, ...ly) - 0.1), hi = Math.ceil(Math.max(...lx, ...ly) + 0.1);
  const sx = v => L + (Math.log10(v) - lo) / (hi - lo) * (W - L - Rr), sy = v => H - B - (Math.log10(v) - lo) / (hi - lo) * (H - B - T);
  const ticks = []; for (let e = lo; e <= hi; e++) ticks.push(10 ** e);
  const tl = v => v >= 1 ? v.toLocaleString() : v.toString();
  const band = (k) => { const a = 10 ** lo, b = 10 ** hi; return `${sx(a)},${sy(Math.min(a * k, b))} ${sx(Math.min(b, b / k))},${sy(Math.min(b, b))}`; };
  const poly = [[10 ** lo, 10 ** lo * 0.2], [10 ** hi, 10 ** hi * 0.2], [10 ** hi, 10 ** hi], [10 ** (hi - Math.log10(5)), 10 ** hi], [10 ** lo, 10 ** lo * 5]]
    .map(([x, y]) => `${sx(x)},${sy(Math.max(Math.min(y, 10 ** hi), 10 ** lo))}`).join(" ");
  const ordered = pts.slice().sort((a, b) => a.bad - b.bad);
  $("#ch-scatter").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="WFP price per kg against Comext EUR per kg">
    <clipPath id="plot"><rect x="${L}" y="${T}" width="${W - L - Rr}" height="${H - B - T}"></rect></clipPath>
    ${ticks.map(t => `<line x1="${sx(t)}" x2="${sx(t)}" y1="${T}" y2="${H - B}" stroke="var(--line)"></line><line x1="${L}" x2="${W - Rr}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--line)"></line>
      <text x="${sx(t)}" y="${H - B + 16}" text-anchor="middle">${tl(t)}</text><text x="${L - 8}" y="${sy(t) + 4}" text-anchor="end">${tl(t)}</text>`).join("")}
    <polygon points="${poly}" fill="var(--s1)" fill-opacity="0.08" clip-path="url(#plot)"></polygon>
    <line x1="${sx(10 ** lo)}" y1="${sy(10 ** lo)}" x2="${sx(10 ** hi)}" y2="${sy(10 ** hi)}" stroke="var(--axis)" clip-path="url(#plot)"></line>
    <line x1="${L}" x2="${W - Rr}" y1="${H - B}" y2="${H - B}" stroke="var(--axis)"></line>
    <text x="${(L + W) / 2}" y="${H - 6}" text-anchor="middle">Comext EUR per kg</text>
    <text transform="translate(12 ${(H - B) / 2}) rotate(-90)" text-anchor="middle">WFP EUR per kg</text>
    <g clip-path="url(#plot)">${ordered.map(p => `<circle data-i="${p.r.i}" cx="${sx(p.x)}" cy="${sy(p.y)}" r="4.5" fill="var(${p.bad ? "--s2" : "--s1"})" stroke="var(--surface)" stroke-width="1.5" style="cursor:pointer"></circle>`).join("")}</g></svg>`;
  const nb = pts.filter(p => p.bad).length;
  $("#lg-scatter").innerHTML = `<span><i style="background:var(--s1);border-radius:50%"></i>Within 0.2-5x <b>${(pts.length - nb).toLocaleString()}</b></span>
    <span><i style="background:var(--s2);border-radius:50%"></i>Price mismatch <b>${nb.toLocaleString()}</b></span><span>Diagonal: same price per kg</span>`;
  $("#ch-scatter").querySelectorAll("circle").forEach(c => {
    const r = R[+c.dataset.i];
    c.addEventListener("mousemove", ev => showTip(ev, `<div class="t">${esc(r.desc)}</div><div class="r">${esc(r.item_std)} · CN ${esc(r.cn)}</div>
      <div class="r">WFP €${fmtN(r.unit_price * m.usd_to_eur / r.kg, 2)}/kg · Comext €${fmtN(r.cx_eur_kg, 2)}/kg · ratio ${r.ratio}</div><div class="r">${fmtKg(r.kg)} per unit · ${fmtUSD(r.value)}</div>`));
    c.addEventListener("mouseleave", hideTip);
    c.addEventListener("click", () => { hideTip(); setFilters({ q: "", item: r.item_id }); go("rows"); openDetail(r.i); });
  });
}

function kgChart() {
  const by = {}; R.forEach(r => { if (r.total_kg > 0) { const k = r.item_std ? r.item_id : "(none)"; by[k] = by[k] || { kg: 0, n: 0, name: r.item_std ? r.item_std.split(",")[0] : k, id: r.item_id }; by[k].kg += r.total_kg; by[k].n++; } });
  const top = Object.values(by).sort((a, b) => b.kg - a.kg).slice(0, 12);
  const W = 600, rowH = 26, L = 190, Rr = 92, H = top.length * rowH + 4, max = top[0]?.kg || 1;
  $("#ch-kg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Estimated tonnes by item">
    ${top.map((t, i) => { const w = t.kg / max * (W - L - Rr), y = i * rowH + 4;
      return `<g data-id="${esc(t.id)}" style="cursor:pointer"><rect x="0" y="${y - 3}" width="${W}" height="${rowH}" fill="transparent"></rect>
      <text x="${L - 10}" y="${y + 13}" text-anchor="end" class="ink2">${esc(t.name.length > 28 ? t.name.slice(0, 27) + "…" : t.name)}</text>
      <rect x="${L}" y="${y + 2}" width="${Math.max(w, 1)}" height="15" rx="${w > 8 ? 4 : 1}" fill="var(--s1)"></rect>
      <text x="${L + w + 8}" y="${y + 13}" class="ink">${fmtN(t.kg / 1000, 0)} t · ${pct(t.kg / totKg)}</text></g>`; }).join("")}</svg>`;
  $("#ch-kg").querySelectorAll("g").forEach(g => {
    const t = top.find(x => x.id === g.dataset.id);
    g.addEventListener("mousemove", ev => showTip(ev, `<div class="t">${esc(t.name)}</div><div class="r">${fmtN(t.kg / 1000, 1)} t across ${t.n} lines</div><div class="r">Click to list rows</div>`));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => { hideTip(); setFilters({ item: t.id, sort: "total_kg" }); go("rows"); });
  });
}

// ---------------------------------------------------------------- checks list
function renderChecks() {
  const order = { integrity: 0, review: 1, info: 2 };
  const label = { integrity: "Integrity", review: "Review", info: "Info" };
  $("#checks").innerHTML = D.checks.slice().sort((a, b) => order[a.severity] - order[b.severity] || b.value_usd - a.value_usd).map(c => {
    const pill = c.severity === "integrity" && !c.rows ? `<span class="pill pass">Pass</span>` : `<span class="pill ${c.severity}">${label[c.severity]}</span>`;
    return `<button type="button" class="check${c.rows ? "" : " zero"}" data-id="${c.id}" ${c.rows ? "" : 'aria-disabled="true"'}>
      <span class="h">${pill}${esc(c.title)}</span><span class="why">${esc(c.why)} <span class="mono">${c.id}</span></span>
      <span class="num"><b>${c.rows.toLocaleString()}</b><span>${c.rows ? pct(c.value_share) + " of value" : "rows"}</span></span></button>`;
  }).join("");
  $("#checks").querySelectorAll(".check").forEach(b => b.addEventListener("click", () => {
    const c = CK[b.dataset.id]; if (!c.rows) return; setFilters({ ck: c.id, sort: "value" }); go("rows"); }));
}

// ---------------------------------------------------------------- rows table
const F = { q: "", ap: "", ck: "", item: "", band: "", max: "", sort: "value", dir: -1, page: 0 };
const PAGE = 100;
const COLS = [
  ["desc", "Description", "desc"], ["item_std", "Standard item"], ["cn", "CN", "code"], ["approach", "Approach"],
  ["kg", "kg / unit", "num"], ["route", "Weight route"], ["units", "Units", "num"], ["total_kg", "Total kg", "num"],
  ["value", "Value USD", "num"], ["unit_price", "USD / unit", "num"], ["score", "Score", "num"], ["checks", "Checks"],
];
function setFilters(p) { Object.assign(F, { q: "", ap: "", ck: "", item: "", band: "", max: "", page: 0 }, p);
  if (p.sort) { F.sort = p.sort; F.dir = -1; } syncControls(); renderRows(); }
function syncControls() { $("#f-q").value = F.q; $("#f-ap").value = F.ap; $("#f-ck").value = F.ck; $("#f-it").value = F.item; $("#f-sc").value = F.max; }
function bandOf(s) { return s >= 70 ? "High" : s >= 40 ? "Medium" : s > 0 ? "Low" : "None"; }
function filtered() {
  const q = F.q.trim().toLowerCase();
  return R.filter(r => (!F.ap || r.approach === F.ap) && (!F.ck || r.checks.includes(F.ck)) && (!F.item || r.item_id === F.item)
    && (!F.band || bandOf(r.score || 0) === F.band) && (F.max === "" || (r.score || 0) <= +F.max)
    && (!q || [r.desc, r.item_std, r.cn, r.item_id, r.country].some(v => String(v ?? "").toLowerCase().includes(q))));
}
function cell(r, k, cls) {
  const v = r[k];
  if (k === "checks") return `<div class="tags">${v.map(c => `<span class="tag ${CK[c].severity}">${c}</span>`).join("")}</div>`;
  if (k === "score") return `<span class="score">${v ?? "–"}</span><span class="bar-s"><i style="width:${v || 0}%"></i></span>`;
  if (k === "approach") return esc(shortAp(v || ""));
  if (k === "kg") return v == null ? "–" : fmtN(v, v < 1 ? 3 : 2);
  if (k === "units") return fmtN(v, 1);
  if (k === "total_kg") return fmtN(v, 0);
  if (k === "value") return fmtN(v, 0);
  if (k === "unit_price") return fmtN(v, 2);
  if (k === "route") return esc(v || (r.kg_basis ? "Research" : "–"));
  return esc(v ?? "–");
}
function renderRows() {
  const rows = filtered();
  const s = F.sort, d = F.dir;
  rows.sort((a, b) => { const x = s === "checks" ? a.checks.length : a[s], y = s === "checks" ? b.checks.length : b[s];
    if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * d; });
  const pages = Math.max(1, Math.ceil(rows.length / PAGE)); F.page = Math.min(F.page, pages - 1);
  const slice = rows.slice(F.page * PAGE, F.page * PAGE + PAGE);
  $("#t-rows thead").innerHTML = "<tr>" + COLS.map(([k, l, c]) => `<th data-k="${k}" class="${c === "num" ? "num" : ""}" ${F.sort === k ? `aria-sort="${d > 0 ? "ascending" : "descending"}"` : ""}>${l}</th>`).join("") + "</tr>";
  $("#t-rows tbody").innerHTML = slice.length ? slice.map(r => `<tr data-i="${r.i}" tabindex="0">${COLS.map(([k, , c]) => `<td class="${c || ""}">${cell(r, k, c)}</td>`).join("")}</tr>`).join("")
    : `<tr><td class="empty" colspan="${COLS.length}">No rows match these filters. Clear a filter to see more.</td></tr>`;
  const val = rows.reduce((a, r) => a + (r.value || 0), 0);
  $("#p-info").textContent = `${rows.length.toLocaleString()} rows · ${fmtUSD(val)} (${pct(val / totalV)} of value) · page ${F.page + 1} of ${pages}`;
  $("#p-prev").disabled = F.page === 0; $("#p-next").disabled = F.page >= pages - 1;
  const act = [F.ap && "Approach: " + shortAp(F.ap), F.ck && "Check: " + CK[F.ck].title, F.item && "Item: " + F.item, F.band && "Band: " + F.band, F.max !== "" && "Score ≤ " + F.max, F.q && `Search: “${F.q}”`].filter(Boolean);
  $("#f-summary").innerHTML = act.map(a => `<span class="tag">${esc(a)}</span>`).join("");
  $("#t-rows thead").querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
    F.dir = F.sort === th.dataset.k ? -F.dir : (["desc", "item_std", "cn", "approach", "route"].includes(th.dataset.k) ? 1 : -1); F.sort = th.dataset.k; renderRows(); }));
  $("#t-rows tbody").querySelectorAll("tr[data-i]").forEach(tr => {
    tr.addEventListener("click", () => openDetail(+tr.dataset.i));
    tr.addEventListener("keydown", e => { if (e.key === "Enter") openDetail(+tr.dataset.i); });
  });
}
function openDetail(i) {
  const r = R[i]; const el = $("#detail"); el.hidden = false;
  document.querySelectorAll("#t-rows tr.sel").forEach(t => t.classList.remove("sel"));
  document.querySelector(`#t-rows tr[data-i="${i}"]`)?.classList.add("sel");
  const dl = pairs => `<dl>${pairs.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  el.innerHTML = `<div class="detail"><h3>${esc(r.desc)}</h3>
    ${dl([["Standard item", esc(r.item_std)], ["Item id", `<span class="mono">${esc(r.item_id)}</span>`], ["Scope / group", esc(r.scope) + " · " + esc(r.group)],
      ["Matched by", esc(r.match) + (r.matched ? ` on “${esc(r.matched)}”` : "")], ["Material group", esc(r.mg)], ["Supplier country", esc(r.country)],
      ["Size attributes", `<span class="mono">${esc(r.attributes || "–")}</span>`]])}
    ${dl([["Approach", esc(r.approach)], ["CN code", `<span class="mono">${esc(r.cn || "–")}</span>`], ["Supplementary unit", esc(r.supp_unit || "–")],
      ["kg per unit", r.kg == null ? "–" : fmtN(r.kg, 4)], ["Weight source", esc(r.kg_basis || "–")],
      ["Comext p25-p75", r.p25 == null ? "–" : `${fmtN(r.p25, 3)} – ${fmtN(r.p75, 3)}`],
      ["Price check", r.ratio == null ? "–" : `WFP ${fmtN(r.unit_price * m.usd_to_eur / r.kg, 2)} vs Comext ${fmtN(r.cx_eur_kg, 2)} EUR/kg (ratio ${r.ratio})`]])}
    ${dl([["Quantity", fmtN(r.qty, 2)], ["Units used", fmtN(r.units, 1) + " · " + esc(r.units_basis)], ["Value", fmtUSD(r.value) + " · " + fmtN(r.unit_price, 2) + " USD per unit"],
      ["Total kg", fmtN(r.total_kg, 1)], ["Confidence", `<b>${r.score}</b> · ${esc(r.conf_basis)}`],
      ["Spend factor", r.ceda_ef == null ? "–" : `${fmtN(r.ceda_ef, 3)} kgCO2e/USD · ${esc(r.ceda_geo)}`],
      ["Checks", r.checks.length ? r.checks.map(c => `<span class="tag ${CK[c].severity}" title="${esc(CK[c].title)}">${c}</span>`).join(" ") : "None"]])}
    <div><button class="btn" type="button" id="d-close">Close</button></div></div>`;
  $("#d-close").addEventListener("click", () => { el.hidden = true; });
  el.scrollIntoView({ block: "nearest" });
}

// ---------------------------------------------------------------- items table
function renderItems() {
  const by = {};
  R.forEach(r => { if (!r.item_id) return; const b = by[r.item_id] = by[r.item_id] || { id: r.item_id, name: (r.item_std || "").split(",")[0], cn: r.cn, ap: {}, n: 0, v: 0, kg: [], upk: [], checks: 0, score: 0 };
    b.n++; b.v += r.value || 0; b.score += (r.score || 0) * (r.value || 0); b.ap[r.approach] = (b.ap[r.approach] || 0) + 1;
    if (r.kg > 0) b.kg.push(r.kg); if (r.kg > 0 && r.unit_price > 0) b.upk.push(r.unit_price / r.kg);
    if (r.checks.some(c => CK[c].severity === "review")) b.checks++; });
  const q = ($("#i-q").value || "").toLowerCase();
  const med = a => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y), h = s.length >> 1; return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2; };
  const qnt = (a, p) => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))]; };
  const items = Object.values(by).filter(b => !q || b.name.toLowerCase().includes(q) || b.id.toLowerCase().includes(q)).map(b => ({ ...b,
    kgm: med(b.kg), kgmin: Math.min(...b.kg), kgmax: Math.max(...b.kg), upkm: med(b.upk), spread: b.upk.length > 2 ? qnt(b.upk, .9) / qnt(b.upk, .1) : null,
    conf: b.v ? b.score / b.v : 0, ap: Object.entries(b.ap).sort((x, y) => y[1] - x[1])[0][0] })).sort((a, b) => b.v - a.v);
  $("#t-items thead").innerHTML = `<tr><th>Item</th><th>CN</th><th>Main approach</th><th class="num">Lines</th><th class="num">Value USD</th><th class="num">kg / unit (median)</th>
    <th class="num">kg range</th><th class="num">USD / kg (median)</th><th class="num">USD/kg spread p90/p10</th><th class="num">Lines to review</th><th class="num">Confidence</th></tr>`;
  $("#t-items tbody").innerHTML = items.map(b => `<tr data-id="${esc(b.id)}" tabindex="0"><td>${esc(b.name)} <span class="mono" style="color:var(--muted)">${esc(b.id)}</span></td>
    <td class="code">${esc(b.cn || "–")}</td><td>${esc(shortAp(b.ap))}</td><td class="num">${b.n}</td><td class="num">${fmtN(b.v, 0)}</td>
    <td class="num">${b.kgm == null ? "–" : fmtN(b.kgm, b.kgm < 1 ? 3 : 1)}</td><td class="num">${b.kg.length ? fmtN(b.kgmin, 2) + " – " + fmtN(b.kgmax, 2) : "–"}</td>
    <td class="num">${b.upkm == null ? "–" : fmtN(b.upkm, 2)}</td>
    <td class="num">${b.spread == null ? "–" : (b.spread > 10 ? `<span class="pill review">${fmtN(b.spread, 0)}x</span>` : fmtN(b.spread, 1) + "x")}</td>
    <td class="num">${b.checks || "–"}</td><td class="num">${b.conf.toFixed(0)}</td></tr>`).join("");
  $("#t-items tbody").querySelectorAll("tr").forEach(tr => {
    const f = () => { setFilters({ item: tr.dataset.id }); go("rows"); };
    tr.addEventListener("click", f); tr.addEventListener("keydown", e => { if (e.key === "Enter") f(); }); });
}

// ---------------------------------------------------------------- tabs + controls
function go(tab) { document.querySelectorAll("main section").forEach(s => s.hidden = s.id !== "tab-" + tab);
  document.querySelectorAll("nav a").forEach(a => a.dataset.tab === tab ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  if (location.hash !== "#" + tab) history.replaceState(null, "", "#" + tab);
  try { localStorage.setItem("nfi-tab", tab); } catch (e) {} }
document.querySelectorAll("nav a").forEach(a => a.addEventListener("click", e => { e.preventDefault(); go(a.dataset.tab); }));
addEventListener("hashchange", () => { const t = location.hash.slice(1); if (["overview", "checks", "rows", "items"].includes(t)) go(t); });
$("#f-ap").innerHTML = `<option value="">All</option>` + APPROACHES.map(a => `<option value="${a}">${shortAp(a)}</option>`).join("");
$("#f-ck").innerHTML = `<option value="">Any</option>` + D.checks.filter(c => c.rows).map(c => `<option value="${c.id}">${esc(c.title)} (${c.rows})</option>`).join("");
const itemIds = [...new Set(R.map(r => r.item_id).filter(Boolean))].sort();
$("#f-it").innerHTML = `<option value="">All</option>` + itemIds.map(i => `<option value="${esc(i)}">${esc(i)}</option>`).join("");
let qT; $("#f-q").addEventListener("input", e => { clearTimeout(qT); qT = setTimeout(() => { F.q = e.target.value; F.page = 0; renderRows(); }, 150); });
$("#f-ap").addEventListener("change", e => { F.ap = e.target.value; F.page = 0; renderRows(); });
$("#f-ck").addEventListener("change", e => { F.ck = e.target.value; F.page = 0; renderRows(); });
$("#f-it").addEventListener("change", e => { F.item = e.target.value; F.page = 0; renderRows(); });
$("#f-sc").addEventListener("input", e => { F.max = e.target.value; F.page = 0; renderRows(); });
$("#f-clear").addEventListener("click", () => setFilters({}));
$("#p-prev").addEventListener("click", () => { F.page--; renderRows(); });
$("#p-next").addEventListener("click", () => { F.page++; renderRows(); });
$("#i-q").addEventListener("input", renderItems);

approachChart(); bandChart(); scatter(); kgChart(); renderChecks(); renderRows(); renderItems();
let start = (location.hash || "").slice(1);
if (!["overview", "checks", "rows", "items"].includes(start)) { try { start = localStorage.getItem("nfi-tab") || "overview"; } catch (e) { start = "overview"; } }
go(["overview", "checks", "rows", "items"].includes(start) ? start : "overview");
</script>
"""

if __name__ == "__main__":
    main()
