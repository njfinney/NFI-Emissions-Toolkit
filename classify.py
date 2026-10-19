#!/usr/bin/env python3
"""
Standardise free-text NFI descriptions, map them to a controlled item list with CN codes,
and estimate unit weights.

Design (why it is consistent):
  * No row-by-row translation. Each item in vocabulary.csv carries multilingual regex
    patterns (EN/FR/ES/PT), so language detection is not needed at all.
  * Rules are applied in file order, first match wins. Every row records which rule and
    which words matched, so any result can be explained and corrected.
  * overrides.csv (raw description -> item_id) is applied before the rules. Use it for the
    long tail and for corrections; it is the place where human/LLM judgement is stored,
    so the same text always gets the same answer.
  * The standard name is item_en (fixed list) plus at most one size attribute.

Usage:
  python classify.py NFI_Categorisations.csv --vocab vocabulary.csv --overrides overrides.csv
Outputs: nfi_classified.csv (row level), nfi_lookup.csv (one row per unique description),
         nfi_item_summary.csv (one row per standard item)
"""
import argparse
import csv
import math
import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------- text normalisation

def norm(s: str) -> str:
    s = re.sub(r"[\u2019\u2018\u00b4`]", " ", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("*", "x").replace("×", "x")
    s = re.sub(r"(\d)\s*[x]\s*(\d)", r"\1x\2", s)          # 4 x 6 -> 4x6
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)               # 14,000 -> 14000
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)                   # 3,50 -> 3.50 (decimal comma)
    s = re.sub(r"[^a-z0-9.]+", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)              # keep only decimal points
    return re.sub(r"\s+", " ", s).strip()


HEAD_SPLIT = re.compile(r"\b(for|pour|para|with|avec|con|com)\b")


def head_of(n: str) -> str:
    """The head noun phrase: 'heater wood burning stove for tent' -> 'heater wood burning stove'."""
    parts = HEAD_SPLIT.split(n, maxsplit=1)
    h = parts[0].strip()
    return h if len(h) >= 3 else n

# ---------------------------------------------------------------- attribute extraction

NUM = r"(\d+(?:\.\d+)?)"
UNIT_LEN = r"(mm|cm|mtrs?|mts?|m|ft|')?"

def attributes(n: str) -> dict:
    a = {}
    m = re.search(NUM + r"\s*" + UNIT_LEN + r"\s*x\s*" + NUM + r"\s*" + UNIT_LEN
                  + r"(?:\s*x\s*" + NUM + r"\s*" + UNIT_LEN + r")?", n)
    if m:
        d1, u1, d2, u2, d3, u3 = m.groups()
        unit = u2 or u1 or u3 or ""
        f = {"mm": 0.001, "cm": 0.01, "ft": 0.3048, "'": 0.3048}.get(unit, 1.0)
        dims = [float(d1), float(d2)] + ([float(d3)] if d3 else [])
        if not unit and max(dims) > 100:   # e.g. 160x180x150 with no unit -> cm
            f = 0.01
        dims = [d * f for d in dims]
        a["dims_m"] = dims
        a["dims_txt"] = "x".join(f"{d:g}" for d in dims) + " m"
    m = re.search(NUM + r"\s*(sq ?m|sqm|sq ?mtrs?|sq ?meters?|m2|mtr2)\b", n)
    if m:
        a["area_m2"] = float(m.group(1))
    elif "dims_m" in a and len(a["dims_m"]) == 2:
        a["area_m2"] = a["dims_m"][0] * a["dims_m"][1]
    m = re.search(NUM + r"\s*(l|lt|lts|ltr|ltrs|litres?|liters?|litros?|litre)\b", n)
    if m:
        a["volume_l"] = float(m.group(1))
    m = re.search(NUM + r"\s*(gallons?|galones|gal)\b", n)
    if m:
        a["volume_l"] = round(float(m.group(1)) * 3.785, 1)
    m = re.search(NUM + r"\s*(kva|kw)\b", n)
    if m:
        a["power"] = f"{float(m.group(1)):g} {m.group(2)}"
    m = re.search(r"\bd\s*=?\s*" + NUM + r"\s*mm\b|" + NUM + r"\s*mm\b", n)
    if m:
        a["diam_mm"] = float(m.group(1) or m.group(2))
    m = re.search(NUM + r"\s*(microns?|micron|mic|mu)\b", n)
    if m:
        a["thick_mm"] = float(m.group(1)) / 1000
    m = re.search(NUM + r"\s*(m|mt|mts|mtr|mtrs|metres?|meters?|metros?)\b(?!\s*x)", n)
    if m and "dims_m" not in a:
        a["length_m"] = float(m.group(1))
    m = re.search(NUM + r"\s*(kg|kgs|kilos?)\b", n)
    if m:
        a["text_kg"] = float(m.group(1))
    m = re.search(NUM + r"\s*(lbs?|libras?|livres?)\b", n)
    if m and "text_kg" not in a:
        a["text_kg"] = round(float(m.group(1)) * 0.4536, 3)
    m = re.search(r"\bqty\s*(\d+)|\b(\d{4,6})\s*(?:pcs|pieces|units|nos)?\s*$|\-\s*(\d{4,6})\b", n)
    if m:
        a["qty_hint"] = float(next(g for g in m.groups() if g))
    return a


def size_label(item_id: str, a: dict) -> str:
    if item_id.startswith(("TARP", "PLASTIC_SHEET", "GEOMEMBRANE")) and "dims_m" in a:
        return a["dims_txt"]
    if item_id.startswith("TENT") and "area_m2" in a:
        return f"{a['area_m2']:g} m2"
    if item_id == "MOSQUITO_NET" and "dims_m" in a:
        return a["dims_txt"]
    if "volume_l" in a:
        return f"{a['volume_l']:g} L"
    if "power" in a:
        return a["power"]
    if item_id in ("PIPE", "VALVE", "PIPE_FITTING") and "diam_mm" in a:
        return f"DN{a['diam_mm']:g}"
    return ""

# ---------------------------------------------------------------- weights

# items where 'N kg' / 'N lb' in the text is the item's own weight (not a capacity)
TEXT_KG_IS_WEIGHT = {"CEMENT", "THREAD", "NAIL", "WIRE_MESH", "PAINT", "TILE_ADHESIVE", "PLASTER",
                     "HAND_TOOL_OTHER", "HAND_TOOL_SHOVEL", "CLEANING", "SAND_GRAVEL", "BEADS", "ROPE"}
TARP_KG_M2 = 0.175    # IFRC 4x6m woven HDPE tarp: 3.9-4.5 kg / 24 m2
HEAVY_TARP_KG_M2 = 0.25  # fumigation / truck sheets: assumption, verify
PE_DENSITY = 920      # kg/m3


SPARK_IGNITION = re.compile(r"\b(inverter|petrol|gasoline|essence|gasolina|portable|portatil)\b")


def generator_kva(a, n):
    """Rated output in kVA from 'N kVA', 'N kW' (kW / 0.8 power factor) or 'N watt(s)'."""
    if "power" in a:
        v, unit = a["power"].split()
        return float(v) if unit == "kva" else float(v) / 0.8
    m = re.search(NUM + r"\s*(watts?|w)\b", n)
    return float(m.group(1)) / 1000 / 0.8 if m else None


def generator_cn(a, n=""):
    """CN8 band for a generating set. Diesel (8502.1x) unless the text says petrol / inverter /
    portable (spark ignition, 8502.20). None when no rating is stated."""
    kva = generator_kva(a, n)
    if kva is None:
        return None
    if SPARK_IGNITION.search(n):
        return "85022020" if kva <= 7.5 else ("85022040" if kva <= 375 else "85022080")
    for top, code in ((7.5, "85021120"), (75, "85021180"), (375, "85021200"),
                      (750, "85021320"), (2000, "85021340")):
        if kva <= top:
            return code
    return "85021380"


# quantity recorded in m2 (confirmed by reviewer), so Comext kg per m2 applies per unit
QTY_IN_M2_ITEMS = {"TILES", "DRYWALL"}


def estimate_kg(item, a):
    """Return (kg_per_unit, basis, confidence) or (None, reason, conf)."""
    iid = item["item_id"]
    dflt = float(item["default_kg"]) if item["default_kg"] else None
    if iid in ("TARPAULIN", "TARP_FUMIGATION", "TARP_TRUCK"):
        rate = TARP_KG_M2 if iid == "TARPAULIN" else HEAVY_TARP_KG_M2
        if "text_kg" in a:
            return a["text_kg"], "weight stated in description", DS["TEXT"]
        if "area_m2" in a:
            if iid == "TARPAULIN":
                return (round(a["area_m2"] * rate, 2),
                        f"Parametric: {a['area_m2']:g} m2 x {rate} kg/m2 (IFRC tarpaulin grammage)", DS["PARAM_SOURCED"])
            return (round(a["area_m2"] * rate, 2),
                    f"Unqualified Estimate: {a['area_m2']:g} m2 x {rate} kg/m2", DS["UNQUALIFIED"])
        if iid == "TARPAULIN":
            return dflt, "Unqualified Estimate: no size given, standard 4x6 m tarpaulin", DS["UNQUALIFIED"]
        return None, "no size given", 0.0
    if iid in ("PLASTIC_SHEET", "GEOMEMBRANE"):
        if "area_m2" in a and "thick_mm" in a:
            kg = a["area_m2"] * a["thick_mm"] / 1000 * PE_DENSITY
            return round(kg, 2), "Parametric: area x thickness x 920 kg/m3 (PE density)", DS["PARAM_FULL"]
        if "area_m2" in a:
            return (round(a["area_m2"] * 0.18, 2),
                    "Unqualified Estimate: area x 0.18 kg/m2 (~200 micron)", DS["UNQUALIFIED"])
        return None, "no size given", 0.0
    if iid == "ROPE" and "diam_mm" in a and "length_m" in a:
        d = a["diam_mm"] / 1000
        kg = 0.7 * 910 * math.pi * d * d / 4 * a["length_m"]
        return round(kg, 2), "Parametric: length x linear mass (PP rope, 70% fill)", DS["PARAM_SOURCED"]
    if iid in TEXT_KG_IS_WEIGHT and "text_kg" in a:
        return a["text_kg"], "weight / pack weight stated in description", DS["TEXT"]
    if dflt is not None:
        cls = item.get("kg_source_class") or "Unqualified Estimate"
        return dflt, item["kg_source"], DS[SOURCE_CLASS[cls]]
    return None, "awaiting Comext kg/unit" if item["weight_method"] == "COMEXT" else (
        item["kg_source"] or "no weight method"), 0.0

# ---------------------------------------------------------------- confidence scoring
# confidence_score = 100 x identification x quantification x quantity   (0-100)
# Each factor comes from a fixed table, so the same evidence always gets the same score.

ID_SCORE = {                    # how sure we are what the item / service is
    "override": 1.0,            # reviewed by hand (overrides.csv)
    "rule:head": 0.9,           # matched on the head noun
    "rule:full": 0.85,          # matched on a keyword anywhere in the text
    "rule:full-fallback": 0.6,  # only matched after the head-noun pass failed
    "material-group-prior": 0.4,  # no usable words; inferred from the material group
}
GENERIC_ID_CAP = 0.6            # item is itself a catch-all (e.g. 'Tent (unspecified)')
GENERIC_ITEMS = {"TENT", "KIT_OTHER", "BUILDING_MISC", "SVC_OTHER", "CRAFT_OTHER"}

DS = {                          # quantification: how good the kg/unit (goods) or sector fit (services) is
    "TEXT": 1.0,                # weight printed in the description
    "CATALOGUE_EXACT": 0.9,     # humanitarian catalogue weight for the same specification
    "PARAM_FULL": 0.85,         # calculated from stated dimensions + known material property
    "PARAM_SOURCED": 0.75,      # stated size x published rate (e.g. IFRC tarp g/m2)
    "CATALOGUE_TYPE": 0.7,      # catalogue weight, but model / type assumed
    "ENGINEERING": 0.7,         # standard pack size, density or tare
    "QTY_IN_KG": 0.7,           # quantity inferred to be in kg from price
    "OVERRIDE_KG": 0.8,         # weight entered by a reviewer
    "COMEXT_MEDIAN": 0.6,       # Comext median kg per supplementary unit (0.4 if p75/p25 > 3)
    "COMEXT_CONVERTED": 0.5,    # Comext median per m / m2 / m3 / L x size parsed from the description
    "COMEXT_WIDE": 0.4,         # (applies to converted values too when p75/p25 > 3)
    "UNQUALIFIED": 0.3,         # Unqualified Estimate
    "PRICE_PROXY": 0.3,         # unit price / Comext EUR per kg
}
SOURCE_CLASS = {"Catalogue (exact)": "CATALOGUE_EXACT", "Catalogue (type assumed)": "CATALOGUE_TYPE",
                "Engineering standard": "ENGINEERING", "Unqualified Estimate": "UNQUALIFIED"}

QTY_FACTOR = {                  # how far the recorded quantity can be trusted
    "ok": 1.0, "stated_in_text": 0.9, "flagged": 0.8, "re_estimated": 0.6, "not_needed": 1.0}

GEO_FACTOR = {"Country": 1.0, "UN subregion": 0.85, "Rest of World": 0.7}  # spend-based only

APPROACHES = ("Average Data (COMEXT)", "Average Data (Research Derived)",
              "Spend-based (OpenCEDA)", "Unallocated")


def band(score):
    return "High" if score >= 70 else "Medium" if score >= 40 else "Low" if score > 0 else "None"

# Price check is only meaningful for fairly homogeneous items. Machinery, IT and mixed
# bundles legitimately vary 10x in price, so they are not flagged.
BULK_KG_ITEMS = {"CEMENT", "PLASTER", "TILE_ADHESIVE"}
NO_PRICE_CHECK_GROUPS = {"Energy", "Electronics", "Agri equipment", "Services", "Out of scope", "Unclear"}
NO_PRICE_CHECK_ITEMS = {"KIT_OTHER", "KIT_RRT", "TENT", "TENT_CLASSROOM", "TENT_MSU", "BUILDING_MISC",
                        "VEHICLE", "PREFAB", "POWER_TOOL", "FREEZER", "OVEN", "KITCHEN_APPLIANCE",
                        "BAKERY_EQUIP", "MILL", "WATER_FILTER", "WATER_TANK", "PUMP", "SCALE",
                        "CONTAINER_SHIPPING", "FLOORING_MODULAR", "INCINERATOR", "MET_INSTRUMENT"}

# ---------------------------------------------------------------- material recipes

def load_recipes(recipes_path, materials_path):
    """item_id -> {ef: composite kg CO2e per kg, summary, use, proxy}. A composite is
    sum(mass share x material factor); shares must add up to 1 for each item."""
    import os
    if not (os.path.exists(recipes_path) and os.path.exists(materials_path)):
        return {}
    mat = pd.read_csv(materials_path).set_index("material_key")
    rec = pd.read_csv(recipes_path)
    unknown = sorted(set(rec.material_key) - set(mat.index))
    if unknown:
        raise SystemExit(f"{recipes_path}: materials missing from {materials_path}: {unknown}")
    out = {}
    for iid, g in rec.groupby("item_id", sort=False):
        if abs(g.mass_share.sum() - 1) > 1e-6:
            raise SystemExit(f"{recipes_path}: shares for {iid} add up to {g.mass_share.sum():g}, not 1")
        f = g.material_key.map(mat.kgco2e_per_kg)
        out[iid] = dict(ef=round(float((g.mass_share * f).sum()), 4),
                        summary=" / ".join(f"{mat.at[m, 'label']} {s:.0%}" for m, s in zip(g.material_key, g.mass_share)),
                        use=g["use"].iat[0], proxy=bool(mat.loc[g.material_key, "is_proxy"].any()))
    return out

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--vocab", default="vocabulary.csv")
    ap.add_argument("--overrides", default="overrides.csv")
    ap.add_argument("--comext", default=None,
                    help="optional output of comext_unit_weights.py (cn8, median_kg_per_unit, eur_per_kg)")
    ap.add_argument("--usd_to_eur", type=float, default=0.90,
                    help="FX used for the price-based proxy (USD unit price -> EUR)")
    ap.add_argument("--prefix", default="nfi")
    ap.add_argument("--ceda_dir", default=".", help="folder with ceda_ef_long.csv etc. (build_ceda_long.py)")
    ap.add_argument("--countries", default="country_iso3.csv")
    ap.add_argument("--recipes", default="recipes.csv", help="item -> material mass shares (long format)")
    ap.add_argument("--materials", default="material_factors.csv", help="build_material_factors.py output")
    args = ap.parse_args()

    vocab = list(csv.DictReader(open(args.vocab, encoding="utf-8")))
    for v in vocab:
        v["rx"] = re.compile(v["pattern"])
    by_id = {v["item_id"]: v for v in vocab}

    overrides = {}
    try:
        for r in csv.DictReader(open(args.overrides, encoding="utf-8-sig")):
            if r.get("item_id"):
                overrides[r["Detailed_Description"].strip()] = r
    except FileNotFoundError:
        pass

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    # ---- classify each unique description once
    # Material group as a last-resort prior (e.g. 'MBARARA' under Tarpaulins = a tarp delivery)
    GROUP_PRIOR = {"Tarpaulins": "TARPAULIN", "Tents": "TENT", "Blankets": "BLANKET"}
    grp_of = df.groupby(df["Detailed_Description"].fillna("").astype(str))[
        "Material_Group_Description"].agg(lambda s: str(s.mode().iat[0]).strip()).to_dict()

    look = []
    for raw in df["Detailed_Description"].fillna("").astype(str).unique():
        n = norm(raw)
        h = head_of(n)
        a = attributes(n)
        hit, how, matched = None, "", ""
        ov = overrides.get(raw.strip())
        if ov and ov["item_id"] in by_id:
            hit, how, matched = by_id[ov["item_id"]], "override", ov.get("note", "")
        else:
            for v in vocab:
                text = n if v["match_on"] == "full" else h
                m = v["rx"].search(text)
                if m:
                    hit, how, matched = v, "rule:" + v["match_on"], m.group(0)
                    break
            if hit is None:  # second chance: any rule on the full text
                for v in vocab:
                    m = v["rx"].search(n)
                    if m:
                        hit, how, matched = v, "rule:full-fallback", m.group(0)
                        break
            if hit is None and grp_of.get(raw) in GROUP_PRIOR:
                hit, how, matched = by_id[GROUP_PRIOR[grp_of[raw]]], "material-group-prior", grp_of[raw]
        rec = {"Detailed_Description": raw, "desc_norm": n}
        if hit is None:
            rec.update(item_id="UNMATCHED", item_en="", item_std="", group="", scope="UNCLEAR",
                       cn="", weight_method="", match="none", matched_text="",
                       kg_per_unit=None, kg_basis="", data_score=0.0, ceda_code="", ceda_target="", ceda_fit=None)
        else:
            cn = hit["cn"]
            if hit["item_id"] == "GENERATOR":
                cn = generator_cn(a, n) or cn
            spend_scope = hit["scope"] in ("WORKS", "SERVICE", "OUT_OF_SCOPE", "ADJUSTMENT", "UNCLEAR")
            kg, basis, conf = (None, "", 0.0) if spend_scope else estimate_kg(hit, a)
            if ov and ov.get("kg_per_unit"):
                kg, basis, conf = float(ov["kg_per_unit"]), ov.get("note") or "reviewer weight", DS["OVERRIDE_KG"]
            size = size_label(hit["item_id"], a)
            rec.update(item_id=hit["item_id"], item_en=hit["item_en"],
                       item_std=(hit["item_en"] + (", " + size if size else "")),
                       group=hit["group"], scope=hit["scope"], cn=cn,
                       weight_method=hit["weight_method"], match=how, matched_text=matched,
                       kg_per_unit=kg, kg_basis=basis, data_score=conf,
                       ceda_code=hit.get("ceda_code", ""), ceda_target=hit.get("ceda_target", ""),
                       ceda_fit=float(hit["ceda_fit"]) if hit.get("ceda_fit") else None,
                       mfg_category=hit.get("mfg_category", ""), mfg_note=hit.get("mfg_note", ""))
        rec["attributes"] = "; ".join(f"{k}={v if not isinstance(v, list) else 'x'.join(f'{d:g}' for d in v)}"
                                      for k, v in a.items() if k != "dims_txt")
        rec["qty_hint"] = a.get("qty_hint")
        look.append(rec)
    lk = pd.DataFrame(look)

    # ---- optional Comext join: (1) kg per supplementary unit, matched to the unit the NFI
    # quantity is in (assumed pieces), (2) EUR per kg for a price proxy
    eur_per_kg = {}
    for c in ("supp_unit", "comext_route", "comext_cn8", "comext_p25", "comext_p75", "comext_eur_per_kg"):
        lk[c] = None
    if args.comext:
        cx = pd.read_csv(args.comext, dtype={"cn8": str})
        if "supp_unit" not in cx:
            raise SystemExit(f"{args.comext} has no supp_unit column: rerun comext_unit_weights.py")

        def cx_pick(cn):
            """CN8 rows under this code that carry a supplementary unit, restricted to the
            unit most of them use. Returns (unit, rows) or (None, rows with no unit)."""
            sub = cx[cx.cn8.str.startswith(str(cn))] if cn else cx.iloc[0:0]
            su = sub[(sub.supp_unit != "NO_SU") & sub.median_kg_per_unit.notna()]
            if su.empty:
                return None, sub
            unit = su.groupby("supp_unit").n_records.sum().idxmax()
            return unit, su[su.supp_unit == unit]

        def attrs_of(s):
            return dict(kv.split("=", 1) for kv in str(s).split("; ") if "=" in kv)

        for i in lk.index:
            cn = lk.at[i, "cn"]
            if not cn or pd.isna(cn):
                continue
            unit, sub = cx_pick(cn)
            if unit is None:
                lk.at[i, "supp_unit"] = "NO_SU" if not sub.empty else None
                continue
            lk.at[i, "supp_unit"] = unit
            lk.at[i, "comext_cn8"] = " ".join(sub.cn8)
            med, p25, p75 = (float(sub[c].median()) for c in ("median_kg_per_unit", "p25", "p75"))
            lk.at[i, "comext_p25"], lk.at[i, "comext_p75"] = p25, p75
            lk.at[i, "comext_eur_per_kg"] = float(sub.eur_per_kg.median())
            # any goods row still without a weight (COMEXT items, and parametric items whose
            # description gave no size) gets the unit-matched Comext value before the price proxy
            if not (pd.isna(lk.at[i, "kg_per_unit"]) and lk.at[i, "scope"] in ("GOODS", "KIT")):
                continue
            wide = p25 > 0 and p75 / p25 > 3
            src = f"CN {cn} ({len(sub)} CN8 in {unit})"
            a = attrs_of(lk.at[i, "attributes"])
            per = {"100_PST": 100, "1000_PST": 1000}
            size = {"M": ("length_m", 1, "m"), "M2": ("area_m2", 1, "m2"),
                    "L": ("volume_l", 1, "L"), "M3": ("volume_l", 0.001, "m3")}
            if lk.at[i, "item_id"] == "GENERATOR" and generator_kva(attrs_of(lk.at[i, "attributes"]),
                                                                     lk.at[i, "desc_norm"]) is None:
                # no rating stated: kg per set varies 100x across bands, use the price proxy
                lk.at[i, "comext_route"] = "NOT_APPLIED_NO_RATING"
                continue
            if unit == "M2" and lk.at[i, "item_id"] in QTY_IN_M2_ITEMS:
                kg, route = med, "COMEXT_CONVERTED"
                basis = f"Comext median kg/m2, quantity recorded in m2 [{route}, M2], {src}"
                score = DS["COMEXT_WIDE"] if wide else DS["COMEXT_CONVERTED"]
            elif unit in ("PST", "PA") or unit in per:
                div = per.get(unit, 1)
                kg = med / div
                route = {"PST": "COMEXT_PST", "PA": "COMEXT_PA"}.get(unit, f"COMEXT_{unit}")
                what = {"PST": "kg per piece", "PA": "kg per pair"}.get(unit, f"kg per {div} pieces / {div}")
                basis, score = f"Comext median {what} [{route}], {src}", DS["COMEXT_WIDE" if wide else "COMEXT_MEDIAN"]
            elif unit in size and size[unit][0] in a:
                key, f, lab = size[unit]
                qty = float(a[key]) * f
                kg = med * qty
                route = "COMEXT_CONVERTED"
                basis = (f"Comext median {med:g} kg/{lab} x {qty:g} {lab} from description "
                         f"[{route}, {unit}], {src}")
                score = DS["COMEXT_WIDE"] if wide else DS["COMEXT_CONVERTED"]
            else:
                # unit needs a size the description does not give (or is not a count, e.g.
                # CEEL): do not apply; the row-level price proxy is used instead
                lk.at[i, "comext_route"] = f"NOT_APPLIED_{unit}"
                continue
            lk.at[i, "kg_per_unit"], lk.at[i, "kg_basis"] = round(kg, 4), basis
            lk.at[i, "data_score"], lk.at[i, "comext_route"] = score, route
        eur_per_kg = {}
        for c in lk.cn.dropna().unique():
            sub = cx[cx.cn8.str.startswith(str(c))].eur_per_kg.dropna() if c else []
            eur_per_kg[c] = float(sub.median()) if len(sub) else None
        # EUR/kg of the same CN8 subset the kg/unit came from, else of the whole code
        lk["comext_eur_per_kg"] = lk.comext_eur_per_kg.where(lk.comext_eur_per_kg.notna(), lk.cn.map(eur_per_kg))

    # ---- row level
    out = df.merge(lk, on="Detailed_Description", how="left")
    q = out["Quantity"].where(out["Quantity"] > 0)
    out["unit_price_usd"] = (out["Goods_Receipt_Value_USD"] / q).round(2)

    # price-based proxy where there is still no kg/unit: unit price / Comext EUR per kg
    if eur_per_kg:
        need = out.kg_per_unit.isna() & out.scope.isin(["GOODS", "KIT"]) & out.cn.notna()
        epk = out.loc[need, "cn"].map(eur_per_kg)
        # unrounded value / quantity: the rounded unit price is off by up to 25% on cheap items
        proxy = (out.loc[need, "Goods_Receipt_Value_USD"] / q[need]) * args.usd_to_eur / epk
        ok = proxy.notna() & (proxy > 0)
        idx = proxy[ok].index
        out.loc[idx, "kg_per_unit"] = proxy[ok].round(6)
        why = (out.loc[idx, "comext_route"].fillna("").str.replace("NOT_APPLIED_NO_RATING", "no power rating", regex=False)
               .str.replace("NOT_APPLIED_", "no size for ", regex=False))
        why = why.where(why != "", out.loc[idx, "supp_unit"].fillna("NO_SU").replace({"NO_SU": "kg only"}))
        out.loc[idx, "kg_basis"] = ("price proxy: unit price x " + f"{args.usd_to_eur:g}" + " / Comext EUR per kg [PRICE_PROXY, "
                                    + why + "], CN " + out.loc[idx, "cn"].astype(str))
        out.loc[idx, "data_score"] = DS["PRICE_PROXY"]
        out.loc[idx, "comext_route"] = "PRICE_PROXY"

    # bulk materials: a 'unit' price far below one bag means the quantity was recorded in kg
    bulk = out["item_id"].isin(BULK_KG_ITEMS) & (out["unit_price_usd"] < 1.5)
    out.loc[bulk, "kg_per_unit"] = 1.0
    out.loc[bulk, "kg_basis"] = "quantity is in kg (unit price too low for a bag)"
    out.loc[bulk, "data_score"] = DS["QTY_IN_KG"]
    out.loc[bulk, "comext_route"] = None

    # Comext price check: WFP price per kg (after FX) vs Comext EUR per kg for the same CN8s.
    # Not meaningful for the price proxy, which equals Comext EUR/kg by construction.
    cx_rows = out["kg_basis"].astype(str).str.startswith("Comext")
    wfp_eur_kg = out["unit_price_usd"] * args.usd_to_eur / out["kg_per_unit"]
    out["comext_price_ratio"] = (wfp_eur_kg / pd.to_numeric(out["comext_eur_per_kg"])).where(cx_rows).round(2)
    PAIR_WORDS = re.compile(r"\b(pairs?|prs?|paires?|pares?|par)\b")

    # quantity sanity check: compare price per kg (or per unit) with the item's own median
    basis = out["unit_price_usd"] / out["kg_per_unit"]
    basis = basis.where(out["kg_per_unit"].notna(), out["unit_price_usd"])
    key = out["item_id"]
    med = basis.groupby(key).transform(lambda s: s[s > 0].median())
    ratio = basis / med
    n_item = key.map(key.value_counts())
    flags = []
    for i, r in out.iterrows():
        f = []
        goods = r["scope"] in ("GOODS", "KIT")
        if goods and r["Quantity"] <= 1 and r["Goods_Receipt_Value_USD"] > 2000:
            f.append("LUMP_SUM")
        elif (goods and n_item[i] >= 3 and pd.notna(ratio[i]) and (ratio[i] > 5 or ratio[i] < 0.2)
              and r["group"] not in NO_PRICE_CHECK_GROUPS and r["item_id"] not in NO_PRICE_CHECK_ITEMS):
            f.append("QTY_UNIT_SUSPECT")
        if goods and r["Quantity"] % 1 != 0 and r["item_id"] not in QTY_IN_M2_ITEMS:
            f.append("QTY_NOT_INTEGER")   # usually m2, m or kg rather than pieces
        cr = r["comext_price_ratio"]
        if pd.notna(cr) and (cr > 5 or cr < 0.2):
            f.append("COMEXT_PRICE_MISMATCH")   # wrong CN8 within the code, or a unit mismatch
        if r["comext_route"] == "COMEXT_PA" and not PAIR_WORDS.search(str(r["desc_norm"])):
            f.append("PAIR_QTY_CHECK")          # kg is per pair; qty may be single items
        if r["item_id"] == "UNMATCHED":
            f.append("UNMATCHED")
        if r["item_id"].startswith("KIT_OTHER") or r["item_id"] in ("TENT", "BUILDING_MISC"):
            f.append("GENERIC_ITEM")
        flags.append("|".join(f))
    out["flags"] = flags
    out["price_ratio_vs_item_median"] = ratio.round(2)

    # units estimate: trust qty unless it is clearly not 'each'. Typical unit price is taken
    # from rows with qty > 1 only, so a lump-sum row never calibrates itself.
    typical_up = out["unit_price_usd"].where(out["Quantity"] > 1).groupby(key).transform("median")
    lump = out["flags"].str.contains("LUMP_SUM") & (out["Goods_Receipt_Value_USD"] > 20 * typical_up)
    pallets = out["flags"].str.contains("QTY_UNIT_SUSPECT") & (out["scope"] == "KIT") & (ratio > 5)
    # far too cheap per kg: quantity is in a smaller unit (m, m2, kg) than the weight assumes
    too_cheap = out["flags"].str.contains("QTY_UNIT_SUSPECT") & (ratio < 0.2) & out["kg_per_unit"].notna()
    pallets = pallets | too_cheap
    hint = out["qty_hint"].where(out["qty_hint"].notna())
    sus = (lump | pallets | (out["flags"].str.contains("LUMP_SUM") & hint.notna())) 
    est = out["Goods_Receipt_Value_USD"] / typical_up
    units = out["Quantity"].astype(float).where(~sus, hint.fillna(est))
    # a rescaled bulk-material row inherits the kg-based unit of its peers
    bulk_rescaled = sus & out["item_id"].isin(BULK_KG_ITEMS) & (typical_up < 1.5) & hint.isna()
    out.loc[bulk_rescaled, "kg_per_unit"] = 1.0
    out.loc[bulk_rescaled, "kg_basis"] = "quantity re-estimated in kg (value / typical price per kg)"
    out.loc[bulk_rescaled, "data_score"] = DS["QTY_IN_KG"]
    out["units_est"] = units.round(1)
    out["units_basis"] = ["qty as recorded" if not s else ("qty stated in text" if pd.notna(h) else
                          "value / item median unit price") for s, h in zip(sus, hint)]
    # price proxy: total kg = value x FX / EUR per kg, so the quantity cancels out. Where units
    # were re-estimated (lump sums), spread that total over the new units instead of
    # multiplying a whole-order 'unit' weight by them again.
    pp = out["kg_basis"].astype(str).str.startswith("price proxy") & (out["units_basis"] != "qty as recorded")
    pp_total = out.loc[pp, "Goods_Receipt_Value_USD"] * args.usd_to_eur / out.loc[pp, "cn"].map(eur_per_kg)
    out.loc[pp, "kg_per_unit"] = (pp_total / out.loc[pp, "units_est"]).round(6)
    out["total_kg_est"] = (out["units_est"] * out["kg_per_unit"]).round(1)

    # ---- proposed emissions calculation approach (applied in this order)
    def approach(r):
        if r["scope"] in ("UNCLEAR", "ADJUSTMENT") or not (r["Goods_Receipt_Value_USD"] > 0):
            return "Unallocated"
        if r["scope"] in ("SERVICE", "WORKS", "OUT_OF_SCOPE"):
            return "Spend-based (OpenCEDA)"
        basis = str(r["kg_basis"])
        if pd.notna(r["kg_per_unit"]) and not basis.startswith(("Comext", "price proxy")):
            return "Average Data (Research Derived)"
        if pd.notna(r["cn"]) and str(r["cn"]).strip():
            return "Average Data (COMEXT)"
        return "Spend-based (OpenCEDA)"          # goods with no weight route at all
    out["emissions_approach"] = out.apply(approach, axis=1)

    # ---- Open CEDA factor: supplier country -> UN subregion average -> Rest of World
    import os
    ef_path = os.path.join(args.ceda_dir, "ceda_ef_long.csv")
    out["ceda_ef_kgco2e_per_usd"] = None
    out["ceda_ef_geography"] = ""
    if os.path.exists(ef_path):
        efc = pd.read_csv(ef_path, dtype={"sector_code": str})
        efr = pd.read_csv(os.path.join(args.ceda_dir, "ceda_regional_ef_long.csv"), dtype={"sector_code": str})
        cmap = pd.read_csv(args.countries)
        iso = dict(zip(cmap.Supplier_Country_Name, cmap.iso3))
        sub = dict(zip(cmap.iso3, cmap.un_subregion))
        f_c = dict(zip(efc.country_code + "|" + efc.sector_code, efc.ef_kgco2e_per_usd_2025_purchaser))
        f_r = dict(zip(efr.un_subregion + "|" + efr.sector_code, efr.ef_kgco2e_per_usd_2025_purchaser))
        efs, geos = [], []
        for cname, code in zip(out["Supplier_Country_Name"], out["ceda_code"]):
            code = "" if pd.isna(code) else str(code)
            if not code:
                efs.append(None); geos.append(""); continue
            c3 = iso.get(str(cname).strip())
            if c3 and f"{c3}|{code}" in f_c:
                efs.append(f_c[f"{c3}|{code}"]); geos.append(f"Country ({c3})")
            elif c3 and f"{sub.get(c3)}|{code}" in f_r:
                efs.append(f_r[f"{sub[c3]}|{code}"]); geos.append(f"UN subregion ({sub[c3]})")
            else:
                efs.append(f_c.get(f"ROW|{code}")); geos.append("Rest of World")
        out["ceda_ef_kgco2e_per_usd"] = efs
        out["ceda_ef_geography"] = geos
    out["spend_kgco2e"] = (out["Goods_Receipt_Value_USD"] * pd.to_numeric(out["ceda_ef_kgco2e_per_usd"])).round(1)
    out.loc[out.emissions_approach == "Unallocated", "spend_kgco2e"] = None

    # ---- production emissions: material recipe (BEIS / ADEME, weight-based) or OpenCEDA (spend-based)
    rc = load_recipes(args.recipes, args.materials)
    for c, k in (("recipe", "summary"), ("composite_ef_kgco2e_per_kg", "ef"), ("recipe_use", "use"),
                 ("recipe_has_proxy", "proxy")):
        out[c] = out["item_id"].map(lambda i: rc.get(i, {}).get(k))
    out["weight_kgco2e"] = (out["total_kg_est"] * out["composite_ef_kgco2e_per_kg"]).round(1)
    tier = out["mfg_category"].fillna("").str[:1]
    has_w = out["weight_kgco2e"].notna() & (out["weight_kgco2e"] > 0) & (out["recipe_use"] == "production")
    basis, note = [], []
    for t, hw, ap, r, mn in zip(tier, has_w, out["emissions_approach"], out["recipe"], out["mfg_note"]):
        if ap == "Unallocated":
            basis.append("Unallocated"); note.append("no value, adjustment or unclear line")
        elif t in ("A", "B") and hw:
            basis.append("Weight-based (material recipe)"); note.append("kg x composite material factor")
        else:
            basis.append("Spend-based (OpenCEDA)")
            note.append({"C": "complex item", "D": "service or works", "X": "outside NFI scope"}.get(t)
                        or ("no open material factor: " + str(mn) if pd.isna(r) else "no weight estimate"))
    out["production_basis"], out["production_basis_note"] = basis, note
    w_based = out["production_basis"].str.startswith("Weight")
    out["production_kgco2e"] = out["weight_kgco2e"].where(w_based, out["spend_kgco2e"])
    out["weight_vs_spend"] = (out["weight_kgco2e"] / out["spend_kgco2e"]).where(out["spend_kgco2e"] > 0).round(2)
    HUBS = {"United Arab Emirates", "Belgium", "Netherlands (Kingdom of the)", "Singapore", "Malaysia", "Panama"}
    hub = (out.emissions_approach == "Spend-based (OpenCEDA)") & out.scope.isin(["GOODS", "KIT"]) & \
        out.Supplier_Country_Name.isin(HUBS)
    out.loc[hub, "flags"] = (out.loc[hub, "flags"].fillna("") + "|SUPPLIER_IN_TRADE_HUB").str.strip("|")

    # ---- confidence score
    def qty_state(r):
        if r["emissions_approach"] in ("Spend-based (OpenCEDA)", "Unallocated"):
            return "not_needed"                  # spend is the activity data
        if str(r["kg_basis"]).startswith("price proxy"):
            return "not_needed"                  # value / EUR-per-kg: quantity cancels out
        if r["units_basis"] == "qty stated in text":
            return "stated_in_text"
        if r["units_basis"] != "qty as recorded":
            return "re_estimated"
        if any(f in str(r["flags"]) for f in ("LUMP_SUM", "QTY_UNIT_SUSPECT", "QTY_NOT_INTEGER")):
            return "flagged"
        return "ok"

    id_s, q_s, d_s, sc, why = [], [], [], [], []
    for _, r in out.iterrows():
        i = ID_SCORE.get(r["match"], 0.0) if r["scope"] != "UNCLEAR" else 0.0
        if r["item_id"] in GENERIC_ITEMS:
            i = min(i, GENERIC_ID_CAP)
        ap = r["emissions_approach"]
        if ap == "Spend-based (OpenCEDA)":
            d = r["ceda_fit"] if pd.notna(r["ceda_fit"]) else 0.0
            d *= GEO_FACTOR.get(str(r["ceda_ef_geography"]).split(" (")[0], 0.0)
        elif ap == "Unallocated":
            d = 0.0
        else:
            d = r["data_score"] if pd.notna(r["data_score"]) else 0.0
        qs = qty_state(r)
        q = QTY_FACTOR[qs]
        score = round(100 * i * d * q)
        id_s.append(i); d_s.append(d); q_s.append(q); sc.append(score)
        why.append(f"ID {i:.2f} x {'sector fit' if ap.startswith('Spend') else 'weight'} {d:.2f} x qty {q:.2f} ({qs})")
    out["id_score"], out["quant_score"], out["qty_factor"] = id_s, d_s, q_s
    out["confidence_score"] = sc
    out["confidence_band"] = [band(x) for x in sc]
    out["confidence_basis"] = why

    cols = ["Material_Group_Description", "Detailed_Description", "item_std", "item_en", "item_id",
            "group", "scope", "emissions_approach", "cn", "ceda_code", "ceda_target",
            "ceda_ef_kgco2e_per_usd", "ceda_ef_geography", "spend_kgco2e", "mfg_category", "mfg_note",
            "recipe", "recipe_use", "recipe_has_proxy", "composite_ef_kgco2e_per_kg", "weight_kgco2e",
            "production_basis", "production_basis_note", "production_kgco2e", "weight_vs_spend", "weight_method",
            "kg_per_unit", "kg_basis", "supp_unit", "comext_route", "comext_cn8", "comext_p25", "comext_p75",
            "comext_eur_per_kg", "comext_price_ratio", "confidence_score", "id_score", "quant_score", "qty_factor", "confidence_band", "confidence_basis",
            "Quantity", "Goods_Receipt_Value_USD", "unit_price_usd", "price_ratio_vs_item_median",
            "flags", "units_est", "units_basis", "total_kg_est", "attributes", "match", "matched_text",
            "Supplier_Country_Name", "Material_Group_Category", "Material_Group_Class",
            "Material_Group_Detailed_Name", "desc_norm"]
    out[cols].to_csv(f"{args.prefix}_classified.csv", index=False)

    lk2 = out.groupby("Detailed_Description").agg(
        rows=("Quantity", "size"), qty=("Quantity", "sum"), value_usd=("Goods_Receipt_Value_USD", "sum"),
        mg=("Material_Group_Description", "first"),
        emissions_approach=("emissions_approach", lambda s: s.mode().iat[0]),
        confidence_score_median=("confidence_score", "median"),
        kg_per_unit=("kg_per_unit", "median"),
        kg_basis=("kg_basis", lambda s: s.mode().iat[0] if s.notna().any() else ""),
        comext_route=("comext_route", lambda s: s.mode().iat[0] if s.notna().any() else ""),
        comext_price_ratio=("comext_price_ratio", "median"),
        flags=("flags", lambda s: "|".join(sorted({f for x in s.fillna("") for f in x.split("|") if f}))),
        recipe=("recipe", "first"), composite_ef_kgco2e_per_kg=("composite_ef_kgco2e_per_kg", "first"),
        production_basis=("production_basis", lambda s: s.mode().iat[0]),
        ceda_ef_geography=("ceda_ef_geography", lambda s: s.mode().iat[0] if s.notna().any() else ""),
        weight_kgco2e=("weight_kgco2e", "sum"), spend_kgco2e=("spend_kgco2e", "sum"),
        production_kgco2e=("production_kgco2e", "sum"),
    ).reset_index().merge(
        lk.drop(columns=["data_score", "ceda_fit", "kg_per_unit", "kg_basis", "comext_route"]),
        on="Detailed_Description")
    lk2.sort_values("value_usd", ascending=False).to_csv(f"{args.prefix}_lookup.csv", index=False)

    summ = out.groupby(["scope", "group", "item_id", "item_en", "emissions_approach", "cn", "ceda_code",
                        "ceda_target"],
                       dropna=False).agg(
        rows=("Quantity", "size"), value_usd=("Goods_Receipt_Value_USD", "sum"),
        units_est=("units_est", "sum"), total_kg_est=("total_kg_est", "sum"),
        kg_per_unit_median=("kg_per_unit", "median"),
        spend_kgco2e=("spend_kgco2e", "sum"), weight_kgco2e=("weight_kgco2e", "sum"),
        production_kgco2e=("production_kgco2e", "sum"),
        mfg_category=("mfg_category", "first"), recipe=("recipe", "first"),
        composite_ef_kgco2e_per_kg=("composite_ef_kgco2e_per_kg", "first"),
        production_basis=("production_basis", lambda s: s.mode().iat[0]),
        confidence_score_median=("confidence_score", "median")).reset_index().sort_values("value_usd", ascending=False)
    summ.to_csv(f"{args.prefix}_item_summary.csv", index=False)

    tot = out["Goods_Receipt_Value_USD"].sum()
    def pct(mask):
        return f"{mask.sum():>5} rows  {out.loc[mask,'Goods_Receipt_Value_USD'].sum()/tot:6.1%} value"
    print("matched     ", pct(out.item_id != "UNMATCHED"))
    print("unmatched   ", pct(out.item_id == "UNMATCHED"))
    for s in ["GOODS", "KIT", "WORKS", "SERVICE", "ADJUSTMENT", "OUT_OF_SCOPE", "UNCLEAR"]:
        print(f"{s:12}", pct(out.scope == s))
    print("has kg/unit ", pct(out.kg_per_unit.notna()))
    print("flagged qty ", pct(out["flags"].str.contains("LUMP_SUM|QTY_UNIT_SUSPECT")))
    for ap in APPROACHES:
        print(f"{ap:32}", pct(out.emissions_approach == ap))
    w = out.Goods_Receipt_Value_USD
    print("value-weighted confidence", round((out.confidence_score * w).sum() / w.sum(), 1))
    for b in ("Weight-based (material recipe)", "Spend-based (OpenCEDA)", "Unallocated"):
        m = out.production_basis == b
        print(f"{b:32}", pct(m), f"{out.loc[m, 'production_kgco2e'].sum() / 1000:9,.0f} t CO2e")


if __name__ == "__main__":
    main()
