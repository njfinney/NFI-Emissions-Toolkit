#!/usr/bin/env python3
"""
Turn Eurostat Comext bulk .dat files into a CSV of kg per unit by CN8 product.

Usage:
  pip install duckdb
  python comext_unit_weights.py "full_v2_2025*.dat" --codes codes.csv --out unit_weights.csv

codes.csv: one entry per line, either "code" or "code,label". Codes can be 4, 6 or 8
digits (dots allowed) and match as prefixes, e.g.
  6306,Tents and tarpaulins
  9404.30,Sleeping bags
  8481,Taps and valves

Output columns: cn8, label, supp_unit, supp_unit_label, cn8_description, n_records,
median_kg_per_unit, p25, p75, weighted_kg_per_unit, eur_per_kg, median_eur_per_kg, total_kg,
total_units
  - supp_unit is the Comext code (PST, PA, 1000_PST, M, M2, M3, L, ..., NO_SU) valid in the
    latest period of the data, from Eurostat COMEXT_METADATA RELATIONS/CN8-SUPPL_UNIT.txt;
    labels come from CLASSIFICATIONS/ENGLISH/SUPPL_UNIT.txt and CN.txt
  - kg_per_unit columns are blank for codes with no supplementary unit (mass-only codes)
  - eur_per_kg is available for every code and feeds the price-based weight proxy
"""
import argparse
import csv
import sys

import duckdb


def load_codes(path):
    codes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            code = row[0].strip().replace(".", "").replace(" ", "")
            if not code.isdigit():  # skip a header line
                continue
            label = row[1].strip() if len(row) > 1 else ""
            codes.append((code, label))
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", help='Path or glob, e.g. "full_v2_2025*.dat"')
    ap.add_argument("--codes", required=True, help="CSV of CN codes (and optional labels)")
    ap.add_argument("--out", default="unit_weights.csv")
    ap.add_argument("--metadata", default="comext_metadata",
                    help="folder with CN8-SUPPL_UNIT.txt, SUPPL_UNIT.txt and CN.txt from "
                         "Eurostat comext/COMEXT_METADATA/CLASSIFICATIONS_AND_RELATIONS")
    args = ap.parse_args()

    codes = load_codes(args.codes)
    if not codes:
        sys.exit("No valid codes found in the codes file.")

    con = duckdb.connect()
    con.execute("CREATE TABLE codes(prefix VARCHAR, label VARCHAR)")
    con.executemany("INSERT INTO codes VALUES (?, ?)", codes)

    path = args.files.replace("'", "''")
    # The full_v2 .dat files are comma-delimited text with a header row.
    con.execute(f"""
        CREATE VIEW raw AS
        SELECT * FROM read_csv('{path}', header=true, delim=',', all_varchar=true,
                               union_by_name=true, ignore_errors=true)
    """)
    cols = [c[0] for c in con.execute("DESCRIBE raw").fetchall()]
    upper = {c.upper().strip(): c for c in cols}
    # Column names differ between Comext releases: full_v2 files use the first name,
    # older extracts the second.
    aliases = {"nc": ("PRODUCT_NC",), "kg": ("QUANTITY_KG", "QUANTITY_IN_KG"),
               "q": ("QUANTITY_SUPPL_UNIT", "SUP_QUANTITY"), "val": ("VALUE_EUR", "VALUE_IN_EUROS")}
    found = {}
    for key, names in aliases.items():
        found[key] = next((upper[n] for n in names if n in upper), None)
        if found[key] is None:
            sys.exit(f"None of {names} found. Columns present: {cols}")
    nc, kg, q, val = found["nc"], found["kg"], found["q"], found["val"]

    # From 2021 the UK appears as GB (whole UK) and as its parts XU (UK excl. Northern
    # Ireland) and XI (Northern Ireland); GB = XU + XI. Keep GB and drop XU/XI for periods
    # that have GB rows, otherwise every UK shipment is counted twice.
    src = "raw"
    if "PARTNER" in upper and "PERIOD" in upper:
        pa, pe = upper["PARTNER"], upper["PERIOD"]
        con.execute(f"""
            CREATE VIEW raw_dedup AS SELECT * FROM raw
            WHERE "{pa}" NOT IN ('XU', 'XI')
               OR "{pe}" NOT IN (SELECT DISTINCT "{pe}" FROM raw WHERE "{pa}" = 'GB')
        """)
        src = "raw_dedup"

    # Supplementary unit per CN8, valid on the first day of the latest period in the data.
    import os
    md = args.metadata.replace("'", "''")
    su_file = os.path.join(args.metadata, "CN8-SUPPL_UNIT.txt")
    if not os.path.exists(su_file):
        sys.exit(f"{su_file} not found. Download it from Eurostat bulk download, folder "
                 "comext/COMEXT_METADATA/CLASSIFICATIONS_AND_RELATIONS/RELATIONS/")
    pe = upper.get("PERIOD")
    ref = con.execute(f'SELECT max("{pe}") FROM {src}').fetchone()[0] if pe else None
    ref_date = f"{ref[:4]}-{ref[4:6]}-01" if ref else "2500-01-01"
    def tsv(name, cols, enc="utf-8"):   # SUPPL_UNIT.txt is Latin-1, the others ASCII
        f = os.path.join(md, name).replace("'", "''")
        return (f"read_csv('{f}', delim='\\t', header=false, all_varchar=true, encoding='{enc}', "
                f"quote='', columns={{{', '.join(f"'{c}': 'VARCHAR'" for c in cols)}}})")
    valid = f"strptime('{ref_date}', '%Y-%m-%d') BETWEEN strptime(d_from, '%d/%m/%Y') AND strptime(d_to, '%d/%m/%Y')"
    con.execute(f"""CREATE TABLE su AS SELECT cn8, any_value(unit) AS unit
                    FROM {tsv('CN8-SUPPL_UNIT.txt', ['cn8', 'unit', 'd_from', 'd_to'])}
                    WHERE {valid} GROUP BY cn8""")
    lab = os.path.join(args.metadata, "SUPPL_UNIT.txt")
    con.execute(f"CREATE TABLE su_lab AS SELECT * FROM {tsv('SUPPL_UNIT.txt', ['unit', 'label'], 'latin-1')}"
                if os.path.exists(lab) else "CREATE TABLE su_lab(unit VARCHAR, label VARCHAR)")
    cnf = os.path.join(args.metadata, "CN.txt")
    con.execute(f"""CREATE TABLE cn_lab AS SELECT cn8, any_value(descr) AS descr
                    FROM {tsv('CN.txt', ['cn8', 'd_from', 'd_to', 'descr'])}
                    WHERE length(cn8) = 8 AND {valid} GROUP BY cn8"""
                if os.path.exists(cnf) else "CREATE TABLE cn_lab(cn8 VARCHAR, descr VARCHAR)")
    print(f"Supplementary units valid on {ref_date}: {con.execute('SELECT count(*) FROM su').fetchone()[0]} CN8 codes")

    query = f"""
    WITH r AS (
      SELECT trim("{nc}") AS cn8,
             try_cast("{kg}" AS DOUBLE) AS kg,
             try_cast("{q}"  AS DOUBLE) AS units,
             try_cast("{val}" AS DOUBLE) AS eur
      FROM {src}
      -- drop TOTAL rows and confidential/aggregated codes such as 8481XXXX
      WHERE regexp_matches(trim("{nc}"), '^[0-9]{{8}}$')
    ),
    m AS (   -- every matching row with a mass: used for EUR per kg
      SELECT r.*, c.label FROM r JOIN codes c ON r.cn8 LIKE c.prefix || '%'
      WHERE r.kg > 0
    ),
    u AS (   -- rows that also have a supplementary quantity: used for kg per unit
      SELECT cn8, kg, units, kg / units AS kg_per_unit FROM m WHERE units > 0
    ),
    us AS (
      SELECT cn8,
             count(*)                                   AS n_records,
             round(median(kg_per_unit), 4)              AS median_kg_per_unit,
             round(quantile_cont(kg_per_unit, 0.25), 4) AS p25,
             round(quantile_cont(kg_per_unit, 0.75), 4) AS p75,
             round(sum(kg) / sum(units), 4)             AS weighted_kg_per_unit,
             round(sum(units), 0)                       AS total_units
      FROM u GROUP BY cn8
    ),
    ms AS (
      SELECT cn8, any_value(label) AS label,
             round(sum(eur) / sum(kg), 3) AS eur_per_kg,
             round(median(eur / kg), 3)   AS median_eur_per_kg,
             round(sum(kg), 0)            AS total_kg
      FROM m WHERE eur > 0 GROUP BY cn8
    )
    SELECT ms.cn8, ms.label,
           coalesce(su.unit, 'NO_SU') AS supp_unit,
           coalesce(su_lab.label, 'No supplementary unit') AS supp_unit_label,
           cn_lab.descr AS cn8_description,
           us.n_records, us.median_kg_per_unit, us.p25, us.p75,
           us.weighted_kg_per_unit, ms.eur_per_kg, ms.median_eur_per_kg, ms.total_kg, us.total_units
    FROM ms LEFT JOIN us USING (cn8)
    LEFT JOIN su USING (cn8)
    LEFT JOIN su_lab ON su_lab.unit = coalesce(su.unit, 'NO_SU')
    LEFT JOIN cn_lab USING (cn8)
    ORDER BY cn8
    """
    con.execute(f"COPY ({query}) TO '{args.out}' (HEADER, DELIMITER ',')")
    n = con.execute(f"SELECT count(*) FROM read_csv('{args.out}')").fetchone()[0]
    print(f"Wrote {n} CN8 codes to {args.out}")
    print("Codes with blank kg/unit are mass-only (no supplementary unit); their eur_per_kg "
          "is still usable for the price proxy.")


if __name__ == "__main__":
    main()
