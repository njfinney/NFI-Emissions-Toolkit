"""Reshape Open CEDA 2025 (wide workbook) into long, filter-friendly tables."""
import openpyxl, pandas as pd, numpy as np, re, sys
src = sys.argv[1] if len(sys.argv) > 1 else "openceda.xlsx"
wb = openpyxl.load_workbook(src, read_only=True, data_only=True)

def sheet(name):
    return [r for r in wb[name].iter_rows(values_only=True)]

# --- sectors
g = sheet("GHG_t")
names = [str(x).strip() for x in g[2][3:] if x is not None]
codes = [str(x).strip() for x in g[3][3:] if x is not None]
n = len(codes)
md = {str(r[0]).strip(): (r[2] or "") for r in sheet("Metadata")[1:] if r[0]}
conv_rows = sheet("Purchaser - producer conversion")
conv = dict(zip([str(c).strip() for c in conv_rows[4][1:n+1]], conv_rows[5][1:n+1]))
pi_rows = sheet("Sector level Price Index")
pi_codes = [str(c).strip() for c in pi_rows[4][1:n+1]]
pi = {str(r[0]): dict(zip(pi_codes, r[1:n+1])) for r in pi_rows[5:] if r[0] and re.fullmatch(r"20\d\d", str(r[0]))}
years = sorted(pi)
sectors = pd.DataFrame({"sector_code": codes, "sector_name": names})
sectors["sector_description"] = sectors.sector_code.map(md)
sectors["purchaser_producer_conversion"] = sectors.sector_code.map(conv).astype(float)
for y in years:
    sectors[f"price_index_{y}"] = sectors.sector_code.map(pi[y]).astype(float)

# --- countries + regions
reg = {r[0]: r[1] for r in sheet("Country to region mapping")[1:] if r[0]}
REGION_FIX = {"Australian and New Zealand": "Australia and New Zealand", "Carribean": "Caribbean",
              "South Eastern Asia": "South-Eastern Asia"}

# --- country EFs: official GHG_t (cached values, 2025 / USD / purchaser) + raw 2023 producer
raw = {r[0]: r for r in sheet("GHG_t_Raw")[4:] if r[0]}
recs = []
for r in g[4:]:
    if not r[0]:
        continue
    cc, cname = r[0], r[1]
    rr = raw[cc]
    for j, code in enumerate(codes):
        recs.append((code, names[j], cc, cname, reg.get(cc, "Rest of World" if cc == "ROW" else ""),
                     r[3 + j], rr[3 + j]))
ef = pd.DataFrame(recs, columns=["sector_code", "sector_name", "country_code", "country_name",
                                 "un_subregion", "ef_kgco2e_per_usd_2025_purchaser",
                                 "ef_kgco2e_per_usd_2023_producer"])
for c in ef.columns[-2:]:
    ef[c] = pd.to_numeric(ef[c], errors="coerce")
ef["lookup_key"] = ef.country_code + "|" + ef.sector_code

# check: rebuild 2025 purchaser from raw with CEDA's own method
chk = ef.merge(sectors[["sector_code", "purchaser_producer_conversion", "price_index_2023", "price_index_2025"]],
               on="sector_code")
rebuilt = chk.ef_kgco2e_per_usd_2023_producer * chk.purchaser_producer_conversion * \
    chk.price_index_2023 / chk.price_index_2025
dev = (rebuilt / chk.ef_kgco2e_per_usd_2025_purchaser - 1).abs()
print("rows", len(ef), "missing official", ef.ef_kgco2e_per_usd_2025_purchaser.isna().sum(),
      "max rel. deviation rebuilt vs official", f"{dev.max():.2e}")

# --- regional EFs (published as 2023 producer; 2025 purchaser derived with the same method)
rg = sheet("Regional Average EFs")
rcodes = [str(x).strip() for x in rg[3][2:] if x is not None]
rrecs = []
for r in rg[4:]:
    if not r[0]:
        continue
    region = REGION_FIX.get(r[0], r[0])
    for j, code in enumerate(rcodes):
        rrecs.append((region, code, r[2 + j]))
reg_ef = pd.DataFrame(rrecs, columns=["un_subregion", "sector_code", "ef_kgco2e_per_usd_2023_producer"])
reg_ef["ef_kgco2e_per_usd_2023_producer"] = pd.to_numeric(reg_ef.ef_kgco2e_per_usd_2023_producer, errors="coerce")
reg_ef = reg_ef.merge(sectors[["sector_code", "sector_name", "purchaser_producer_conversion",
                               "price_index_2023", "price_index_2025"]], on="sector_code", how="left")
reg_ef["ef_kgco2e_per_usd_2025_purchaser"] = (reg_ef.ef_kgco2e_per_usd_2023_producer *
    reg_ef.purchaser_producer_conversion * reg_ef.price_index_2023 / reg_ef.price_index_2025)
reg_ef = reg_ef[["un_subregion", "sector_code", "sector_name", "ef_kgco2e_per_usd_2025_purchaser",
                 "ef_kgco2e_per_usd_2023_producer"]]
reg_ef["lookup_key"] = reg_ef.un_subregion + "|" + reg_ef.sector_code
print("regional rows", len(reg_ef), "regions", reg_ef.un_subregion.nunique(),
      "unmatched sector codes", reg_ef.sector_name.isna().sum())

countries = pd.DataFrame(sorted(reg.items()), columns=["country_code", "un_subregion"])
in_ceda = set(ef.country_code)
countries["in_ceda_country_list"] = countries.country_code.isin(in_ceda)
countries = countries.merge(ef[["country_code", "country_name"]].drop_duplicates(), on="country_code", how="left")
missing_regions = set(countries.un_subregion) - set(reg_ef.un_subregion)
print("subregions with no regional EF:", missing_regions)

sectors.to_csv("ceda_sectors.csv", index=False)
ef.to_csv("ceda_ef_long.csv", index=False)
reg_ef.to_csv("ceda_regional_ef_long.csv", index=False)
countries.to_csv("ceda_countries.csv", index=False)
