"""Gene descriptions and PubMed counts for the panel genes."""
import time
import xml.etree.ElementTree as ET
from datetime import date

import pandas as pd
import requests

from common import RES, retry

panel = pd.read_csv(RES / "panels.csv")
panel_of = dict(zip(panel["gene"], panel["panel"]))

# ---------- gene names and descriptions (exact symbol first, alias only for old names)
from common import resolve_symbols
resolved = resolve_symbols(list(panel_of))
rows = [{"gene_in_data": g, "panel": panel_of[g], "current_symbol": v["symbol"],
         "full_name": v["name"], "mapping_note": v["note"], "summary": v["summary"]}
        for g, v in resolved.items()]
table = pd.DataFrame(rows)
table.to_csv(RES / "gene_lookup.csv", index=False)
print(table[["gene_in_data", "current_symbol", "full_name", "mapping_note"]].to_string(index=False))
print("\nCHECK BY HAND: every row marked OLD NAME or NOT FOUND (is the new name really the same gene?)")

# ---------- PubMed
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
GB = '(gallbladder[tiab] OR "gall bladder"[tiab] OR "gallbladder neoplasms"[MeSH])'
CA = '(cancer[tiab] OR carcinoma[tiab] OR tumor[tiab] OR tumour[tiab] OR "neoplasms"[MeSH])'
BIL = ('("biliary tract"[tiab] OR cholangiocarcinoma[tiab] OR "bile duct"[tiab] '
       'OR "biliary tract neoplasms"[MeSH])')


def esearch(term, retmax=6):
    def _go():
        r = requests.get(BASE + "esearch.fcgi", params={"db": "pubmed", "term": term,
                         "retmax": retmax, "retmode": "json"}, timeout=30).json()
        time.sleep(0.4)
        return int(r["esearchresult"]["count"]), r["esearchresult"]["idlist"]
    return retry(_go, label="pubmed search")


def fetch(ids):
    if not ids:
        return []
    def _go():
        x = requests.get(BASE + "efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids),
                         "retmode": "xml"}, timeout=60).text
        time.sleep(0.4)
        return x
    out = []
    for a in ET.fromstring(retry(_go, label="pubmed fetch")).findall(".//PubmedArticle"):
        t = a.find(".//ArticleTitle")
        out.append((a.findtext(".//PMID"), "".join(t.itertext()) if t is not None else "",
                    a.findtext(".//Journal/Title") or "",
                    a.findtext(".//PubDate/Year") or a.findtext(".//PubDate/MedlineDate") or "",
                    " ".join("".join(x.itertext()) for x in a.findall(".//AbstractText"))))
    return out


def gene_term(row):
    s = {str(row["gene_in_data"]), str(row["current_symbol"])} - {"NOT FOUND", "nan"}
    return "(" + " OR ".join(f'"{x}"[tiab]' for x in sorted(s)) + ")"


lit = []
for _, r in table.iterrows():
    gt = gene_term(r)
    n_gb, ids_gb = esearch(f"{gt} AND {GB}")
    n_ca, _ = esearch(f"{gt} AND {CA}")
    n_bil, ids_bil = (esearch(f"{gt} AND {BIL}") if n_gb == 0 else (0, []))
    lit.append({"gene_in_data": r["gene_in_data"], "current_symbol": r["current_symbol"],
                "panel": r["panel"], "gallbladder_hits": n_gb, "cancer_hits": n_ca,
                "biliary_hits_if_no_gallbladder": n_bil, "gb_ids": " ".join(ids_gb),
                "bil_ids": " ".join(ids_bil)})
    print(f"{r['gene_in_data']:>12}: {n_gb} gallbladder | {n_ca} cancer | {n_bil} biliary")
lit = pd.DataFrame(lit)
lit["search_date"] = date.today().isoformat()
lit["my_decision"] = ""
lit["PMIDs_read"] = ""
lit.drop(columns=["gb_ids", "bil_ids"]).to_csv(RES / "literature_check.csv", index=False)

# ---------- reading list: only genes that need human judgement
lines = []
for _, r in lit.iterrows():
    for label, ids in (("GALLBLADDER", r["gb_ids"]), ("BILIARY", r["bil_ids"])):
        arts = fetch(str(ids).split()) if str(ids).strip() not in ("", "nan") else []
        if not arts:
            continue
        lines.append(f"\n{'=' * 70}\n{r['gene_in_data']} [{label}, {len(arts)} papers]\n{'=' * 70}")
        for pmid, title, jour, year, abst in arts:
            lines.append(f"\nPMID {pmid} | {year} | {jour}\nhttps://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
                         f"TITLE: {title}\nABSTRACT: {abst or '(no abstract)'}")
(RES / "reading_list.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nsaved gene_lookup.csv, literature_check.csv, reading_list.txt in results/")
print("Next: read reading_list.txt and fill my_decision in literature_check.csv.")
