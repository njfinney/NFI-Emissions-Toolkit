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
    tier = o["mfg_category"].fillna("").str[:1]
    wb = o["production_basis"].fillna("").str.startswith("Weight")
    chosen = o["weight_kgco2e"].where(wb, o["spend_kgco2e"])
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
        ("PRODUCTION_BASIS", "integrity", "Production basis contradicts the manufacturing category",
         "Weight-based needs category A or B, a production recipe and a weight; the total must equal the chosen method.",
         lambda: (wb & (~tier.isin(["A", "B"]) | o.composite_ef_kgco2e_per_kg.isna() | ~(o.total_kg_est > 0)))
         | (o.production_kgco2e.notna() & ((o.production_kgco2e - chosen).abs() > 0.5))),
        ("WEIGHT_SPEND_GAP", "review", "Weight-based and spend-based differ more than 5x",
         "Usually a wrong recipe, a wrong weight or a wrong spend sector.",
         lambda: wb & ((o.weight_vs_spend > 5) | (o.weight_vs_spend < 0.2))),
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
        ("MATERIAL_PROXY", "info", "Weight-based with a labelled proxy material factor",
         "See material_factors.csv for the proxy and its direction (over- or understates).",
         lambda: wb & o.recipe_has_proxy.fillna(False).astype(bool)),
        ("TRADE_HUB", "info", "Spend-based goods bought from a re-export hub",
         "The supplier country may not be where the goods were made.", lambda: flag(o, "SUPPLIER_IN_TRADE_HUB")),
    ]


def build_tree(o, recipes_path="recipes.csv", materials_path="material_factors.csv"):
    """Leaves for the emissions breakdown: one per (item, material component) for weight-based
    rows, one per item for spend-based rows. kg CO2e of a component = item kg x share x factor."""
    import os
    if not (os.path.exists(recipes_path) and os.path.exists(materials_path)):
        return []
    mat = pd.read_csv(materials_path).set_index("material_key")
    rec = pd.read_csv(recipes_path)
    rec = rec[rec["use"] == "production"]
    wb = o["production_basis"].fillna("").str.startswith("Weight")
    leaves = []
    for iid, g in o[wb].groupby("item_id"):
        kg = float(g.total_kg_est.sum())
        for _, x in rec[rec.item_id == iid].iterrows():
            m = mat.loc[x.material_key]
            leaves.append(dict(tier=g.mfg_category.iat[0], item=g.item_en.iat[0], item_id=iid, comp=m.label,
                               basis="weight", co2=round(kg * x.mass_share * m.kgco2e_per_kg, 1),
                               kg=round(kg * x.mass_share, 1), share=x.mass_share, ef=m.kgco2e_per_kg,
                               src=f"{m.source}: {m.source_detail}", proxy=bool(m.is_proxy), lines=len(g),
                               value=float(g.Goods_Receipt_Value_USD.sum())))
    sp = o[~wb & o.production_kgco2e.notna() & (o.production_kgco2e > 0)]
    for (iid, sector), g in sp.groupby(["item_id", "ceda_target"]):
        leaves.append(dict(tier=g.mfg_category.iat[0], item=g.item_en.iat[0], item_id=iid,
                           comp=f"Spend-based: {sector}", basis="spend", co2=round(float(g.production_kgco2e.sum()), 1),
                           kg=None, share=None, ef=None, src="OpenCEDA 2025 by supplier country", proxy=False,
                           lines=len(g), value=float(g.Goods_Receipt_Value_USD.sum())))
    return leaves


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
            "ceda_ef": "ceda_ef_kgco2e_per_usd", "ceda_geo": "ceda_ef_geography", "spend_co2": "spend_kgco2e",
            "ceda_code": "ceda_code", "ceda_target": "ceda_target", "tier": "mfg_category", "mfg_note": "mfg_note",
            "recipe": "recipe", "cef": "composite_ef_kgco2e_per_kg", "w_co2": "weight_kgco2e",
            "p_basis": "production_basis", "p_note": "production_basis_note", "p_co2": "production_kgco2e",
            "wvs": "weight_vs_spend"}
    keys = list(cols)
    rows = [[clean(r[c]) for c in cols.values()] + [row_checks[i]] for i, r in o.iterrows()]
    meta = dict(generated=dt.datetime.now().strftime("%d %b %Y %H:%M"), comext_period=args.comext_period,
                usd_to_eur=args.usd_to_eur, rows=len(o), source=f"{args.prefix}_classified.csv")
    tree = build_tree(o)
    data = json.dumps(dict(keys=keys + ["checks"], rows=rows, checks=checks, meta=meta, tree=tree),
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
select { max-width: min(260px, 100%) }
.filters label { min-width: 0; max-width: 100% }
main .wrap > section { min-width: 0 }
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
.em-head { display: flex; flex-wrap: wrap; gap: 12px 24px; justify-content: space-between; align-items: start }
.em-head > div { min-width: 0; max-width: 680px }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; flex-shrink: 0 }
.seg button { font: 600 13px var(--font); color: var(--ink-2); background: var(--raise); border: 0; padding: 7px 12px; cursor: pointer }
.seg button + button { border-left: 1px solid var(--line) }
.seg button[aria-pressed="true"] { background: var(--accent-wash); color: var(--accent) }
.crumbs { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin: 4px 0 10px; font-size: 13px }
.crumbs button { font: 600 13px var(--font); color: var(--accent); background: none; border: 0; padding: 2px 4px; cursor: pointer; border-radius: 4px }
.crumbs button:hover { background: var(--accent-wash) }
.crumbs span.sep { color: var(--muted) }
.crumbs span.cur { font-weight: 700; color: var(--ink); padding: 2px 4px }
#em-map svg { display: block; width: 100%; height: auto }
#em-map .tile { cursor: pointer }
#em-map .tile.leaf { cursor: default }
#em-map .tile:focus-visible rect.bg { stroke: var(--accent); stroke-width: 2 }
#em-map text.tl { fill: var(--ink); font: 600 12px var(--font) }
#em-map text.tv { fill: var(--ink-2); font: 11.5px var(--font); font-variant-numeric: tabular-nums }
.proxy { font-size: 11px; padding: 0 6px; border-radius: 3px; background: var(--warn-wash); color: var(--warn); white-space: nowrap }
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
    <a href="#emissions" data-tab="emissions">Emissions</a>
  </nav>
</div></header>

<main><div class="wrap">
<section id="tab-overview">
  <div class="kpis" id="kpis"></div>
  <div class="grid2" style="margin-top:16px">
    <div class="panel"><h2>Value by emissions approach</h2><p class="sub">Share of Goods Receipt value (USD)</p><div id="ch-approach"></div><div class="legend" id="lg-approach"></div></div>
    <div class="panel"><h2>Value by confidence band</h2><p class="sub">High 70+ · Medium 40-69 · Low 1-39 · None 0</p><div id="ch-band"></div></div>
    <div class="panel"><h2>WFP price per kg against Comext</h2><p class="sub">COMEXT rows with a Comext kg/unit, EUR per kg, log scales. Outside the shaded 0.2-5x band is flagged. Click a point to open the row.</p><div id="ch-scatter"></div><div class="legend" id="lg-scatter"></div></div>
    <div class="panel"><h2>Production emissions by manufacturing category</h2><p class="sub">t CO2e, cradle-to-gate. Weight-based = kg x composite BEIS / ADEME material factor; spend-based = OpenCEDA by supplier country.</p><div id="ch-tier"></div><div class="legend" id="lg-tier"></div></div>
    <div class="panel"><h2>Weight-based against spend-based</h2><p class="sub">Weight-based rows: both methods in t CO2e, by item, top 12 by weight-based emissions. Bars far apart point to a recipe, weight or sector to check.</p><div id="ch-wvs"></div><div class="legend" id="lg-wvs"></div></div>
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
    <label for="f-ti">Category<select id="f-ti"></select></label>
    <label for="f-sc">Max score<input type="number" id="f-sc" min="0" max="100" step="5" style="width:90px"></label>
    <button class="btn" id="f-clear" type="button">Clear filters</button>
  </div>
  <div class="active-filter" id="f-summary"></div>
  <div class="panel" id="detail" hidden></div>
  <div class="scroll"><table id="t-rows"><thead></thead><tbody></tbody></table></div>
  <div class="pager"><span id="p-info"></span><span><button class="btn" id="p-prev" type="button">Previous</button> <button class="btn" id="p-next" type="button">Next</button></span></div>
</section>

<section id="tab-emissions" hidden>
  <div class="panel">
    <div class="em-head">
      <div><h2>Where production emissions come from</h2>
      <p class="sub">Tile area = t CO2e. Select a tile to open it; use the trail to go back. Weight-based items split into their recipe components (item kg x mass share x material factor); spend-based items show their OpenCEDA sector.</p></div>
      <div class="seg" role="group" aria-label="Group by">
        <button type="button" id="em-by-item" aria-pressed="true">Category, item, material</button>
        <button type="button" id="em-by-mat" aria-pressed="false">Material, item</button>
      </div>
    </div>
    <nav class="crumbs" id="em-crumbs" aria-label="Breakdown level"></nav>
    <div id="em-map"></div>
    <div class="legend" id="em-legend"></div>
  </div>
  <div class="scroll" style="margin-top:16px"><table id="t-em"><thead></thead><tbody></tbody></table></div>
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
const totCo2 = R.reduce((a, r) => a + (r.p_co2 || 0), 0);
const isW = r => (r.p_basis || "").startsWith("Weight");
const wbCo2 = R.filter(isW).reduce((a, r) => a + (r.p_co2 || 0), 0);
const TIERS = ["A. Simple", "B. Multi-material", "C. Complex", "D. Services and works", "X. Unallocated / out of scope"];
const integ = D.checks.filter(c => c.severity === "integrity");
const integBad = integ.filter(c => c.rows > 0);
const review = D.checks.filter(c => c.severity === "review");
const reviewRows = new Set(R.filter(r => r.checks.some(c => CK[c].severity === "review")).map(r => r.i));
$("#kpis").innerHTML = [
  ["Goods receipt value", fmtUSD(totalV), `${R.length.toLocaleString()} lines`],
  ["Confidence, value-weighted", wConf.toFixed(1), "out of 100"],
  ["Value with a kg per unit", pct(withKg), "goods, kits and services"],
  ["Estimated weight", fmtN(totKg / 1000, 0) + " t", "sum of total_kg_est"],
  ["Production emissions", fmtN(totCo2 / 1000, 0) + " t CO2e", `${pct(wbCo2 / totCo2)} weight-based`],
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

function tierChart() {
  const W = 600, rowH = 36, L = 150, Rr = 120, H = TIERS.length * rowH + 4;
  const d = TIERS.map(t => { const rs = R.filter(r => r.tier === t);
    return { t, w: rs.filter(isW).reduce((a, r) => a + (r.p_co2 || 0), 0), s: rs.filter(r => !isW(r)).reduce((a, r) => a + (r.p_co2 || 0), 0), n: rs.length }; });
  const max = Math.max(...d.map(x => x.w + x.s)) || 1, sc = v => v / max * (W - L - Rr);
  $("#ch-tier").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Production emissions by manufacturing category">
    ${d.map((x, i) => { const y = i * rowH + 6, ww = sc(x.w), sw = sc(x.s), g = ww > 0 && sw > 0 ? 2 : 0;
      return `<g data-t="${esc(x.t)}" style="cursor:pointer"><rect x="0" y="${y - 4}" width="${W}" height="${rowH}" fill="transparent"></rect>
      <text x="${L - 10}" y="${y + 15}" text-anchor="end" class="ink2">${esc(x.t.replace(" / out of scope", ""))}</text>
      ${ww > 0 ? `<rect x="${L}" y="${y + 3}" width="${Math.max(ww, 1)}" height="18" rx="${ww > 8 ? 4 : 1}" fill="var(--s1)"></rect>` : ""}
      ${sw > 0 ? `<rect x="${L + ww + g}" y="${y + 3}" width="${Math.max(sw - g, 1)}" height="18" rx="${sw > 8 ? 4 : 1}" fill="var(--s2)"></rect>` : ""}
      <text x="${L + ww + sw + 8}" y="${y + 16}" class="ink">${fmtN((x.w + x.s) / 1000, 0)} t · ${pct((x.w + x.s) / totCo2)}</text></g>`; }).join("")}</svg>`;
  $("#lg-tier").innerHTML = `<span><i style="background:var(--s1)"></i>Weight-based <b>${fmtN(wbCo2 / 1000, 0)} t</b></span><span><i style="background:var(--s2)"></i>Spend-based <b>${fmtN((totCo2 - wbCo2) / 1000, 0)} t</b></span>`;
  $("#ch-tier").querySelectorAll("g").forEach(g => { const x = d.find(v => v.t === g.dataset.t);
    g.addEventListener("mousemove", ev => showTip(ev, `<div class="t">${esc(x.t)}</div><div class="r">Weight-based ${fmtN(x.w / 1000, 1)} t · spend-based ${fmtN(x.s / 1000, 1)} t</div><div class="r">${x.n} lines · click to list rows</div>`));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => { hideTip(); setFilters({ tier: x.t, sort: "p_co2" }); go("rows"); }); });
}

function wvsChart() {
  const by = {}; R.filter(isW).forEach(r => { const b = by[r.item_id] = by[r.item_id] || { id: r.item_id, name: (r.item_std || "").split(",")[0], w: 0, s: 0 }; b.w += r.w_co2 || 0; b.s += r.spend_co2 || 0; });
  const top = Object.values(by).sort((a, b) => b.w - a.w).slice(0, 12);
  const W = 600, rowH = 30, L = 170, Rr = 110, H = top.length * rowH + 4, max = Math.max(...top.map(t => Math.max(t.w, t.s))) || 1, sc = v => v / max * (W - L - Rr);
  $("#ch-wvs").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Weight-based against spend-based emissions by item">
    ${top.map((t, i) => { const y = i * rowH + 4;
      return `<g data-id="${esc(t.id)}" style="cursor:pointer"><rect x="0" y="${y - 2}" width="${W}" height="${rowH}" fill="transparent"></rect>
      <text x="${L - 10}" y="${y + 15}" text-anchor="end" class="ink2">${esc(t.name.length > 26 ? t.name.slice(0, 25) + "…" : t.name)}</text>
      <rect x="${L}" y="${y + 2}" width="${Math.max(sc(t.w), 1)}" height="10" rx="3" fill="var(--s1)"></rect>
      <rect x="${L}" y="${y + 14}" width="${Math.max(sc(t.s), 1)}" height="10" rx="3" fill="var(--s2)"></rect>
      <text x="${L + Math.max(sc(t.w), sc(t.s)) + 8}" y="${y + 17}" class="ink">${t.s ? (t.w / t.s).toFixed(1) + "x" : "–"}</text></g>`; }).join("")}</svg>`;
  $("#lg-wvs").innerHTML = `<span><i style="background:var(--s1)"></i>Weight-based</span><span><i style="background:var(--s2)"></i>Spend-based (OpenCEDA)</span><span>Label: weight ÷ spend</span>`;
  $("#ch-wvs").querySelectorAll("g").forEach(g => { const t = top.find(x => x.id === g.dataset.id);
    g.addEventListener("mousemove", ev => showTip(ev, `<div class="t">${esc(t.name)}</div><div class="r">Weight-based ${fmtN(t.w / 1000, 1)} t · spend-based ${fmtN(t.s / 1000, 1)} t</div>`));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => { hideTip(); setFilters({ item: t.id, sort: "p_co2" }); go("rows"); }); });
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
const F = { q: "", ap: "", ck: "", item: "", tier: "", band: "", max: "", sort: "value", dir: -1, page: 0 };
const PAGE = 100;
const COLS = [
  ["desc", "Description", "desc"], ["item_std", "Standard item"], ["cn", "CN", "code"], ["approach", "Approach"],
  ["kg", "kg / unit", "num"], ["route", "Weight route"], ["units", "Units", "num"], ["total_kg", "Total kg", "num"],
  ["value", "Value USD", "num"], ["unit_price", "USD / unit", "num"], ["tier", "Category"], ["p_co2", "kg CO2e", "num"],
  ["p_basis", "Basis"], ["score", "Score", "num"], ["checks", "Checks"],
];
function setFilters(p) { Object.assign(F, { q: "", ap: "", ck: "", item: "", tier: "", band: "", max: "", page: 0 }, p);
  if (p.sort) { F.sort = p.sort; F.dir = -1; } syncControls(); renderRows(); }
function syncControls() { $("#f-q").value = F.q; $("#f-ap").value = F.ap; $("#f-ck").value = F.ck; $("#f-it").value = F.item; $("#f-ti").value = F.tier; $("#f-sc").value = F.max; }
function bandOf(s) { return s >= 70 ? "High" : s >= 40 ? "Medium" : s > 0 ? "Low" : "None"; }
function filtered() {
  const q = F.q.trim().toLowerCase();
  return R.filter(r => (!F.ap || r.approach === F.ap) && (!F.ck || r.checks.includes(F.ck)) && (!F.item || r.item_id === F.item) && (!F.tier || r.tier === F.tier)
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
  if (k === "p_co2") return fmtN(v, 0);
  if (k === "tier") return esc((v || "–").split(".")[0]);
  if (k === "p_basis") return esc((v || "–").split(" ")[0]);
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
  const act = [F.ap && "Approach: " + shortAp(F.ap), F.ck && "Check: " + CK[F.ck].title, F.item && "Item: " + F.item, F.tier && "Category: " + F.tier, F.band && "Band: " + F.band, F.max !== "" && "Score ≤ " + F.max, F.q && `Search: “${F.q}”`].filter(Boolean);
  $("#f-summary").innerHTML = act.map(a => `<span class="tag">${esc(a)}</span>`).join("");
  $("#t-rows thead").querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
    F.dir = F.sort === th.dataset.k ? -F.dir : (["desc", "item_std", "cn", "approach", "route", "tier", "p_basis"].includes(th.dataset.k) ? 1 : -1); F.sort = th.dataset.k; renderRows(); }));
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
    ${dl([["Manufacturing category", esc(r.tier || "–") + (r.mfg_note ? " · " + esc(r.mfg_note) : "")],
      ["Material recipe", esc(r.recipe || "–")], ["Composite factor", r.cef == null ? "–" : fmtN(r.cef, 3) + " kg CO2e per kg"],
      ["Weight-based", r.w_co2 == null ? "–" : fmtN(r.w_co2, 1) + " kg CO2e"],
      ["Spend-based", r.spend_co2 == null ? "–" : `${fmtN(r.spend_co2, 1)} kg CO2e · ${esc(r.ceda_code)} ${esc(r.ceda_target)}`],
      ["Basis used", `<b>${esc(r.p_basis || "–")}</b> · ${esc(r.p_note || "")}`],
      ["Production emissions", r.p_co2 == null ? "–" : `<b>${fmtN(r.p_co2, 1)} kg CO2e</b>` + (r.wvs != null ? ` · weight ÷ spend ${r.wvs}` : "")]])}
    <div><button class="btn" type="button" id="d-close">Close</button></div></div>`;
  $("#d-close").addEventListener("click", () => { el.hidden = true; });
  el.scrollIntoView({ block: "nearest" });
}

// ---------------------------------------------------------------- items table
function renderItems() {
  const by = {};
  R.forEach(r => { if (!r.item_id) return; const b = by[r.item_id] = by[r.item_id] || { id: r.item_id, name: (r.item_std || "").split(",")[0], cn: r.cn, tier: r.tier, cef: r.cef, ap: {}, n: 0, v: 0, kg: [], upk: [], checks: 0, score: 0, co2: 0, wc: 0, sc: 0 };
    b.co2 += r.p_co2 || 0; if (isW(r)) { b.wc += r.w_co2 || 0; b.sc += r.spend_co2 || 0; }
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
    <th class="num">kg range</th><th class="num">USD / kg (median)</th><th class="num">USD/kg spread p90/p10</th><th class="num">Lines to review</th><th class="num">Confidence</th><th>Category</th><th class="num">kg CO2e / kg</th><th class="num">t CO2e</th><th class="num">Weight ÷ spend</th></tr>`;
  $("#t-items tbody").innerHTML = items.map(b => `<tr data-id="${esc(b.id)}" tabindex="0"><td>${esc(b.name)} <span class="mono" style="color:var(--muted)">${esc(b.id)}</span></td>
    <td class="code">${esc(b.cn || "–")}</td><td>${esc(shortAp(b.ap))}</td><td class="num">${b.n}</td><td class="num">${fmtN(b.v, 0)}</td>
    <td class="num">${b.kgm == null ? "–" : fmtN(b.kgm, b.kgm < 1 ? 3 : 1)}</td><td class="num">${b.kg.length ? fmtN(b.kgmin, 2) + " – " + fmtN(b.kgmax, 2) : "–"}</td>
    <td class="num">${b.upkm == null ? "–" : fmtN(b.upkm, 2)}</td>
    <td class="num">${b.spread == null ? "–" : (b.spread > 10 ? `<span class="pill review">${fmtN(b.spread, 0)}x</span>` : fmtN(b.spread, 1) + "x")}</td>
    <td class="num">${b.checks || "–"}</td><td class="num">${b.conf.toFixed(0)}</td><td>${esc((b.tier || "–").split(".")[0])}</td>
    <td class="num">${b.cef == null ? "–" : fmtN(b.cef, 2)}</td><td class="num">${fmtN(b.co2 / 1000, 1)}</td>
    <td class="num">${b.sc > 0 ? ((b.wc / b.sc) > 5 || (b.wc / b.sc) < 0.2 ? `<span class="pill review">${(b.wc / b.sc).toFixed(1)}x</span>` : (b.wc / b.sc).toFixed(1) + "x") : "–"}</td></tr>`).join("");
  $("#t-items tbody").querySelectorAll("tr").forEach(tr => {
    const f = () => { setFilters({ item: tr.dataset.id }); go("rows"); };
    tr.addEventListener("click", f); tr.addEventListener("keydown", e => { if (e.key === "Enter") f(); }); });
}

// ---------------------------------------------------------------- emissions breakdown (click-through treemap)
const TIER_COLOR = { "A. Simple": "--s1", "B. Multi-material": "--s2", "C. Complex": "--s3", "D. Services and works": "--s4", "X. Unallocated / out of scope": "--axis" };
let EM = { by: "item", path: [] };
function emTree() {
  const root = { name: "All production emissions", children: [] };
  const get = (node, name, extra) => { let c = node.children.find(x => x.name === name); if (!c) { c = Object.assign({ name, children: [] }, extra); node.children.push(c); } return c; };
  D.tree.forEach(l => {
    const leaf = { name: l.comp, value: l.co2, leaf: l };
    if (EM.by === "item") get(get(root, l.tier, { tier: l.tier }), l.item, { tier: l.tier, item_id: l.item_id }).children.push(leaf);
    else get(get(root, l.basis === "spend" ? "Spend-based (no material split)" : l.comp, { mat: true, spend: l.basis === "spend" }), l.item, { tier: l.tier, item_id: l.item_id })
      .children.push(Object.assign({}, leaf, { name: l.basis === "spend" ? l.comp.replace("Spend-based: ", "") : l.item }));
  });
  const sum = n => n.children ? (n.value = n.children.reduce((a, c) => a + sum(c), 0)) : n.value;
  sum(root);
  const sort = n => { if (n.children) { n.children.sort((a, b) => b.value - a.value); n.children.forEach(sort); } };
  sort(root);
  return root;
}
function emColor(n, parent) {
  const t = n.tier || n.leaf?.tier || parent?.tier;
  if (EM.by === "mat" && EM.path.length === 0) return n.spend ? "--s2" : "--s1";
  return TIER_COLOR[t] || "--s1";
}
function squarify(items, x, y, w, h) {
  // squarified treemap layout (Bruls et al.), items sorted descending by value
  const out = [], total = items.reduce((a, c) => a + c.value, 0) || 1, area = w * h;
  let rest = items.map(c => ({ c, a: c.value / total * area })), rx = x, ry = y, rw = w, rh = h;
  const worst = (row, side) => { const s = row.reduce((a, r) => a + r.a, 0), mx = Math.max(...row.map(r => r.a)), mn = Math.min(...row.map(r => r.a));
    return Math.max(side * side * mx / (s * s), (s * s) / (side * side * mn)); };
  while (rest.length) {
    const side = Math.min(rw, rh); let row = [rest[0]], i = 1;
    while (i < rest.length && worst(row.concat(rest[i]), side) <= worst(row, side)) { row.push(rest[i]); i++; }
    rest = rest.slice(i);
    const s = row.reduce((a, r) => a + r.a, 0);
    if (rw >= rh) { const cw = s / rh; let cy = ry; row.forEach(r => { const ch = r.a / cw; out.push({ c: r.c, x: rx, y: cy, w: cw, h: ch }); cy += ch; }); rx += cw; rw -= cw; }
    else { const ch = s / rw; let cx = rx; row.forEach(r => { const cw = r.a / ch; out.push({ c: r.c, x: cx, y: ry, w: cw, h: ch }); cx += cw; }); ry += ch; rh -= ch; }
  }
  return out;
}
function renderEmissions() {
  const root = emTree();
  let node = root; const trail = [root];
  for (const name of EM.path) { const n = node.children?.find(c => c.name === name); if (!n) { EM.path = EM.path.slice(0, trail.length - 1); break; } node = n; trail.push(n); }
  $("#em-crumbs").innerHTML = trail.map((n, i) => i === trail.length - 1 ? `<span class="cur">${esc(n.name)}</span>`
    : `<button type="button" data-depth="${i}">${esc(n.name)}</button><span class="sep">›</span>`).join("");
  $("#em-crumbs").querySelectorAll("button").forEach(b => b.addEventListener("click", () => { EM.path = EM.path.slice(0, +b.dataset.depth); renderEmissions(); }));
  const kids = (node.children || []).filter(c => c.value > 0);
  const W = Math.max(280, Math.round($("#em-map").clientWidth || 1000)), H = Math.round(Math.min(460, Math.max(300, W * 0.55))), gap = 2;
  const tiles = squarify(kids, 0, 0, W, H);
  const fmtT = v => v >= 1e4 ? fmtN(v / 1000, 0) + " t" : v >= 1000 ? fmtN(v / 1000, 1) + " t" : fmtN(v, 0) + " kg";
  $("#em-map").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Production emissions breakdown: ${esc(node.name)}">
    ${tiles.map((t, i) => { const c = t.c, col = emColor(c, node), x = t.x + gap / 2, y = t.y + gap / 2, w = Math.max(t.w - gap, 0), h = Math.max(t.h - gap, 0);
      const big = w > 90 && h > 40, mid = w > 56 && h > 22, name = c.name.length > Math.floor(w / 7.2) ? c.name.slice(0, Math.max(Math.floor(w / 7.2) - 1, 3)) + "…" : c.name;
      return `<g class="tile${c.children ? "" : " leaf"}" data-i="${i}" tabindex="${c.children ? 0 : -1}" role="${c.children ? "button" : "img"}" aria-label="${esc(c.name)}: ${fmtT(c.value)}">
        <rect class="bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="color-mix(in srgb, var(${col}) ${c.leaf?.basis === "spend" ? 12 : 24}%, var(--surface))"></rect>
        <rect x="${x}" y="${y}" width="${w}" height="${Math.min(4, h)}" rx="2" fill="var(${col})"></rect>
        ${mid ? `<text class="tl" x="${x + 8}" y="${y + 20}">${esc(name)}</text>` : ""}
        ${big ? `<text class="tv" x="${x + 8}" y="${y + 37}">${fmtT(c.value)} · ${pct(c.value / root.value)}</text>` : ""}</g>`; }).join("")}</svg>`;
  $("#em-map").querySelectorAll(".tile").forEach(g => { const c = tiles[+g.dataset.i].c;
    const tipHtml = () => `<div class="t">${esc(c.name)}</div><div class="r">${fmtT(c.value)} CO2e · ${pct(c.value / node.value)} of ${esc(node.name)} · ${pct(c.value / root.value)} of all</div>`
      + (c.leaf && c.leaf.basis === "weight" ? `<div class="r">${fmtN(c.leaf.kg / 1000, 1)} t material × ${fmtN(c.leaf.ef, 3)} kg CO2e/kg${c.leaf.proxy ? " · proxy factor" : ""}</div>` : "")
      + (c.children ? `<div class="r">Select to open (${c.children.length})</div>` : "");
    g.addEventListener("mousemove", ev => showTip(ev, tipHtml())); g.addEventListener("mouseleave", hideTip);
    const open = () => { if (!c.children) return; hideTip(); EM.path = EM.path.concat(c.name); renderEmissions(); };
    g.addEventListener("click", open); g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } }); });
  const usedTiers = [...new Set(D.tree.map(l => l.tier))];
  $("#em-legend").innerHTML = EM.by === "mat" && EM.path.length === 0
    ? `<span><i style="background:var(--s1)"></i>Material (weight-based)</span><span><i style="background:var(--s2)"></i>Spend-based</span>`
    : TIERS.filter(t => usedTiers.includes(t)).map(t => `<span><i style="background:var(${TIER_COLOR[t]})"></i>${esc(t)}</span>`).join("") + `<span>Paler tile: spend-based</span>`;
  // grid of the current level
  const isLeafLevel = kids.length && !kids[0].children;
  $("#t-em thead").innerHTML = `<tr><th>${esc(isLeafLevel ? (EM.by === "item" ? "Component" : "Item") : "Name")}</th><th class="num">t CO2e</th><th class="num">Share of ${esc(node.name.length > 24 ? "level" : node.name)}</th><th class="num">Share of all</th>
    ${isLeafLevel ? `<th class="num">Mass share</th><th class="num">Material t</th><th class="num">kg CO2e / kg</th><th>Factor source</th>` : `<th class="num">Parts</th>`}</tr>`;
  $("#t-em tbody").innerHTML = kids.map((c, i) => `<tr data-i="${i}" ${c.children ? 'tabindex="0"' : ""}><td>${esc(c.name)}${c.leaf?.proxy ? ' <span class="proxy">proxy</span>' : ""}</td>
    <td class="num">${fmtN(c.value / 1000, 1)}</td><td class="num">${pct(c.value / node.value)}</td><td class="num">${pct(c.value / root.value)}</td>
    ${isLeafLevel ? (c.leaf.basis === "weight" ? `<td class="num">${pct(c.leaf.share)}</td><td class="num">${fmtN(c.leaf.kg / 1000, 1)}</td><td class="num">${fmtN(c.leaf.ef, 3)}</td><td>${esc(c.leaf.src)}</td>`
      : `<td class="num">–</td><td class="num">–</td><td class="num">–</td><td>${esc(c.leaf.src)}</td>`) : `<td class="num">${c.children.length}</td>`}</tr>`).join("");
  $("#t-em tbody").querySelectorAll("tr[tabindex]").forEach(tr => { const c = kids[+tr.dataset.i];
    const open = () => { EM.path = EM.path.concat(c.name); renderEmissions(); };
    tr.addEventListener("click", open); tr.addEventListener("keydown", e => { if (e.key === "Enter") open(); }); });
}
let emT; addEventListener("resize", () => { clearTimeout(emT); emT = setTimeout(() => { if (!$("#tab-emissions").hidden) renderEmissions(); }, 150); });
$("#em-by-item").addEventListener("click", () => { EM = { by: "item", path: [] }; $("#em-by-item").setAttribute("aria-pressed", "true"); $("#em-by-mat").setAttribute("aria-pressed", "false"); renderEmissions(); });
$("#em-by-mat").addEventListener("click", () => { EM = { by: "mat", path: [] }; $("#em-by-mat").setAttribute("aria-pressed", "true"); $("#em-by-item").setAttribute("aria-pressed", "false"); renderEmissions(); });

// ---------------------------------------------------------------- tabs + controls
function go(tab) { document.querySelectorAll("main section").forEach(s => s.hidden = s.id !== "tab-" + tab);
  if (tab === "emissions") renderEmissions();
  document.querySelectorAll("nav a").forEach(a => a.dataset.tab === tab ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  if (location.hash !== "#" + tab) history.replaceState(null, "", "#" + tab);
  try { localStorage.setItem("nfi-tab", tab); } catch (e) {} }
document.querySelectorAll("nav a").forEach(a => a.addEventListener("click", e => { e.preventDefault(); go(a.dataset.tab); }));
addEventListener("hashchange", () => { const t = location.hash.slice(1); if (["overview", "checks", "rows", "items", "emissions"].includes(t)) go(t); });
$("#f-ap").innerHTML = `<option value="">All</option>` + APPROACHES.map(a => `<option value="${a}">${shortAp(a)}</option>`).join("");
$("#f-ck").innerHTML = `<option value="">Any</option>` + D.checks.filter(c => c.rows).map(c => `<option value="${c.id}">${esc(c.title)} (${c.rows})</option>`).join("");
const itemIds = [...new Set(R.map(r => r.item_id).filter(Boolean))].sort();
$("#f-it").innerHTML = `<option value="">All</option>` + itemIds.map(i => `<option value="${esc(i)}">${esc(i)}</option>`).join("");
let qT; $("#f-q").addEventListener("input", e => { clearTimeout(qT); qT = setTimeout(() => { F.q = e.target.value; F.page = 0; renderRows(); }, 150); });
$("#f-ap").addEventListener("change", e => { F.ap = e.target.value; F.page = 0; renderRows(); });
$("#f-ck").addEventListener("change", e => { F.ck = e.target.value; F.page = 0; renderRows(); });
$("#f-it").addEventListener("change", e => { F.item = e.target.value; F.page = 0; renderRows(); });
$("#f-ti").innerHTML = `<option value="">All</option>` + TIERS.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join("");
$("#f-ti").addEventListener("change", e => { F.tier = e.target.value; F.page = 0; renderRows(); });
$("#f-sc").addEventListener("input", e => { F.max = e.target.value; F.page = 0; renderRows(); });
$("#f-clear").addEventListener("click", () => setFilters({}));
$("#p-prev").addEventListener("click", () => { F.page--; renderRows(); });
$("#p-next").addEventListener("click", () => { F.page++; renderRows(); });
$("#i-q").addEventListener("input", renderItems);

approachChart(); bandChart(); scatter(); tierChart(); wvsChart(); kgChart(); renderChecks(); renderRows(); renderItems(); renderEmissions();
let start = (location.hash || "").slice(1);
if (!["overview", "checks", "rows", "items", "emissions"].includes(start)) { try { start = localStorage.getItem("nfi-tab") || "overview"; } catch (e) { start = "overview"; } }
go(["overview", "checks", "rows", "items", "emissions"].includes(start) ? start : "overview");
</script>
"""

if __name__ == "__main__":
    main()
