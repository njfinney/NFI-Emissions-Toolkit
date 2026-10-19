#!/usr/bin/env python3
"""
Build material_factors.csv (kg CO2e per kg, cradle-to-gate, primary material) from the published
sources named in material_spec.csv.

Sources (in factor_sources/):
  ghg-conversion-factors-2025-full-set.xlsx  UK Government (DESNZ/BEIS) GHG conversion factors 2025,
                                             sheet 'Material use', column 'Primary material production'
  base_carbone.csv                           ADEME Base Carbone (Licence Ouverte), from
                                             https://data.ademe.fr/datasets/base-carboner (make factors)
Usage: python build_material_factors.py
"""
import csv
import sys

import openpyxl

SRC = "factor_sources"


def beis_material_use(path):
    """{(activity, material): kg CO2e per tonne, primary material production}"""
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["Material use"]
    out, activity = {}, None
    for r in ws.iter_rows(values_only=True):
        a, m, unit, primary = (r + (None,) * 4)[:4]
        if a == "Activity":
            continue
        if a and m and unit == "tonnes":
            activity = str(a).strip()
        if m and unit == "tonnes" and activity:
            out[(activity, str(m).strip())] = primary
    return out


def ademe(path):
    """{id: (name, kg CO2e per kg, status)} for valid mass-based emission factors."""
    text = open(path, encoding="latin-1", newline="").read().split("\r\n")
    head = text[0].split(";")
    ix = {h: i for i, h in enumerate(head)}
    out = {}
    for line in text[1:]:
        p = line.split(";")
        if len(p) != len(head) or p[ix["Type Ligne"]] != "Elément":
            continue
        unit = p[ix["Unité français"]]
        if unit not in ("kgCO2e/tonne", "kgCO2e/kg"):
            continue
        v = float(p[ix["Total poste non décomposé"]].replace(",", "."))
        name = " | ".join(x for x in (p[ix["Nom base anglais"]], p[ix["Nom attribut anglais"]],
                                      p[ix["Nom frontière anglais"]]) if x)
        out[p[ix["Identifiant de l'élément"]]] = (name, v / (1000 if unit == "kgCO2e/tonne" else 1),
                                                  p[ix["Statut de l'élément"]])
    return out


def main():
    beis = beis_material_use(f"{SRC}/ghg-conversion-factors-2025-full-set.xlsx")
    bc = ademe(f"{SRC}/base_carbone.csv")
    rows, errors = [], []
    for s in csv.DictReader(open("material_spec.csv", encoding="utf-8")):
        if s["source"] == "BEIS 2025":
            _, act, mat = s["source_ref"].split("|")
            v = beis.get((act, mat))
            if v is None:
                errors.append(f"{s['material_key']}: BEIS row not found: {act} / {mat}")
                continue
            kg, name = float(v) / 1000, f"Material use: {act} / {mat} (primary material production)"
        else:
            hit = bc.get(s["source_ref"])
            if hit is None or not hit[2].startswith("Valide"):
                errors.append(f"{s['material_key']}: Base Carbone id {s['source_ref']} not found or not valid")
                continue
            kg, name = hit[1], f"Base Carbone id {s['source_ref']}: {hit[0]}"
        rows.append(dict(material_key=s["material_key"], label=s["label"], kgco2e_per_kg=round(kg, 5),
                         source=s["source"], source_detail=name, is_proxy="Proxy" in s["proxy_note"],
                         note=s["proxy_note"], boundary="cradle-to-gate, primary material"))
    if errors:
        sys.exit("\n".join(errors))
    with open("material_factors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote material_factors.csv: {len(rows)} materials")
    for r in rows:
        print(f"  {r['material_key']:18} {r['kgco2e_per_kg']:8.3f}  {r['source']:18} {'proxy' if r['is_proxy'] else ''}")


if __name__ == "__main__":
    main()
