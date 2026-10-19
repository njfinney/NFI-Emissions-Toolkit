"""Regression tests for the NFI toolkit. Run from nfi_toolkit/: python -m pytest -q tests"""
import csv
import os
import re
import subprocess
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import classify as C  # noqa: E402


def path(*p):
    return os.path.join(HERE, *p)


VOCAB = list(csv.DictReader(open(path("vocabulary.csv"), encoding="utf-8")))

# ---------------------------------------------------------------- vocabulary hygiene


def test_patterns_compile_and_ids_unique():
    ids = [v["item_id"] for v in VOCAB]
    assert len(ids) == len(set(ids))
    for v in VOCAB:
        re.compile(v["pattern"])


def test_every_cn_is_extracted_from_comext():
    codes = [r["code"] for r in csv.DictReader(open(path("codes_nfi.csv"), encoding="utf-8-sig"))]
    for v in VOCAB:
        cn = v["cn"]
        if cn and not cn.startswith("3004"):          # medicines are out of scope
            assert any(cn.startswith(c) or c.startswith(cn) for c in codes), f"{v['item_id']} {cn} not in codes_nfi.csv"


@pytest.mark.skipif(not os.path.exists(path("comext_metadata", "CN.txt")), reason="no CN metadata")
def test_every_cn_exists_in_current_cn_year():
    """Catches codes retired at a CN revision (e.g. 441810 -> 441811/441819 in 2022)."""
    valid = set()
    for row in csv.reader(open(path("comext_metadata", "CN.txt"), encoding="utf-8"), delimiter="\t"):
        if len(row) >= 3 and row[2].endswith("2500"):
            valid.add(row[0])
    bad = [(v["item_id"], v["cn"]) for v in VOCAB if v["cn"] and v["cn"] not in valid and not v["cn"].startswith("3004")]
    assert not bad, f"CN codes not valid in the current CN year: {bad}"


# ---------------------------------------------------------------- manufacturing category and recipes

TIERS = {"A. Simple", "B. Multi-material", "C. Complex", "D. Services and works", "X. Unallocated / out of scope"}


def test_every_item_has_a_category_and_a_spend_sector():
    for v in VOCAB:
        assert v["mfg_category"] in TIERS, v["item_id"]
        if v["mfg_category"][0] != "X":
            assert v["ceda_code"], f"{v['item_id']} has no OpenCEDA sector"


def test_recipes_are_complete_and_sum_to_one():
    rec = pd.read_csv(path("recipes.csv"))
    mats = set(pd.read_csv(path("material_factors.csv")).material_key)
    assert set(rec.material_key) <= mats
    sums = rec.groupby("item_id").mass_share.sum()
    assert ((sums - 1).abs() < 1e-6).all(), sums[(sums - 1).abs() >= 1e-6]
    ids = {v["item_id"] for v in VOCAB}
    assert set(rec.item_id) <= ids
    assert rec.share_basis.notna().all()


def test_material_factors_trace_to_a_source():
    m = pd.read_csv(path("material_factors.csv"))
    assert m.source.isin(["BEIS 2025", "ADEME Base Carbone"]).all()
    assert (m.kgco2e_per_kg > 0).all() and m.source_detail.notna().all()


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize("text,kva_code", [
    ("744 kVA prime Genset", "85021320"), ("230 kVA prime Genset", "85021200"),
    ("Generator, 12.5kva with trailer", "85021180"), ("Generator 5 kVA", "85021120"),
    ("2200 watt Inverter Generator", "85022020"), ("Generator 10 kW petrol", "85022040"),
    ("Electricity Generator", None),
])
def test_generator_bands(text, kva_code):
    n = C.norm(text)
    assert C.generator_cn(C.attributes(n), n) == kva_code


@pytest.mark.parametrize("text,key,value", [
    ("Used Portafloor Stock Dubai 710 SqMtr", "area_m2", 710), ("Tarpaulin 4x6 m", "area_m2", 24),
    ("Plastic Basin 40 Litres", "volume_l", 40), ("Cement 50kg bag", "text_kg", 50),
])
def test_attributes(text, key, value):
    assert C.attributes(C.norm(text))[key] == pytest.approx(value)


# ---------------------------------------------------------------- classification (golden cases)
# description -> expected item_id. Add a line whenever a reviewer fixes a mapping.
GOLDEN = {
    "Cement, Portland M400, 50kg bag": "CEMENT", "Cimento (Sacos de 50kg)": "CEMENT",
    "Poly Sacks  Size 50kg": "SACK", "Biscicletas": "BICYCLE", "MOTO COCIMECAM AK47 avec lunette et casq": "MOTORCYCLE",
    "Trailer for 45KVA gener +fixing": "VEHICLE", "Brouette": "WHEELBARROW", "Aluminium wash basin": "BASIN_ALU",
    "PLASTIC BASIN 40 LITRES": "BASIN", "Manufacture of Washbasins for schools": "SINK",
    "High Filtration Respirators KN95 Masks w": "PPE", "Tapones auditivos industriales": "PPE",
    "MEGAPHONES": "MEGAPHONE", "Loudspeakers": "AUDIO", "Whistle 115 DB": "WHISTLE",
    "Heater, wood burning stove, for tent": "STOVE_HEATER", "Portable gas stove": "STOVE_GAS",
    "Hybrid/Solar generator": "INVERTER", "Tarpaulins 4x6 M, CIF Khartoum": "TARPAULIN",
}


@pytest.fixture(scope="module")
def classified(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    rows = [dict(Material_Group_Category="", Material_Group_Class="", Material_Group_Detailed_Name="",
                 Material_Group_Description="Oth field equipment", Detailed_Description=t, Quantity=10, Goods_Receipt_Value_USD=100,
                 Supplier_Country_Name="Kenya") for t in GOLDEN]
    pd.DataFrame(rows).to_csv(d / "in.csv", index=False)
    subprocess.run([sys.executable, path("classify.py"), str(d / "in.csv"), "--vocab", path("vocabulary.csv"),
                    "--overrides", path("overrides.csv"), "--comext", path("unit_weights.csv"),
                    "--ceda_dir", HERE, "--countries", path("country_iso3.csv"), "--prefix", str(d / "t")],
                   check=True, capture_output=True)
    return pd.read_csv(d / "t_classified.csv")


@pytest.mark.parametrize("text", list(GOLDEN))
def test_golden_mapping(classified, text):
    got = classified.loc[classified.Detailed_Description == text, "item_id"].iloc[0]
    assert got == GOLDEN[text]


# ---------------------------------------------------------------- output integrity


@pytest.mark.skipif(not os.path.exists(path("nfi_classified.csv")), reason="no pipeline output yet")
def test_integrity_checks_pass():
    sys.path.insert(0, HERE)
    import build_review as B
    o = pd.read_csv(path("nfi_classified.csv"), dtype={"cn": str})
    cx = pd.read_csv(path("unit_weights.csv"), dtype={"cn8": str})
    epk = {c: cx[cx.cn8.str.startswith(str(c))].eur_per_kg.median() for c in o.cn.dropna().unique()}
    fx = float(re.search(r"price proxy: unit price x ([\d.]+)", " ".join(o.kg_basis.dropna().astype(str))).group(1))
    for cid, sev, title, _, fn in B.build_checks(o, epk, fx):
        if sev == "integrity":
            n = int(pd.Series(fn(), index=o.index).fillna(False).astype(bool).sum())
            assert n == 0, f"{cid}: {title} ({n} rows)"
