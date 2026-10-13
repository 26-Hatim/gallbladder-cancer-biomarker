"""Download and clean the four GEO studies."""
import argparse
import re

import numpy as np
import pandas as pd

from common import PROC, RAW, RES, MIN_GENE_READS, check, download, get_geo, retry, save_study

FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"


# ---------------------------------------------------------------- GSE139682
def clean_GSE139682():
    """China, RNA-seq (RPKM), 10 tumor + 10 matched normal."""
    url = f"{FTP}/GSE139nnn/GSE139682/suppl/GSE139682_all.rpkm.txt.gz"
    df = pd.read_csv(download(url, RAW / "GSE139682_all.rpkm.txt.gz"), sep="\t")
    expr = df.set_index("GeneID").groupby(level=0).mean()
    X = np.log2(expr.T + 1)                       # samples x genes, log2
    X = X.loc[:, (X > 1).sum() >= 5]              # drop genes that are almost never active
    group = np.where(X.index.str.startswith("Tumor"), "tumor", "normal")
    patient = pd.Series(X.index).str.extract(r"(\d+)")[0].astype(int).values
    labels = pd.DataFrame({"group": group, "patient": patient}, index=X.index)
    check("samples x genes", X.shape, (20, 19183))
    save_study("GSE139682", X, labels)


# ---------------------------------------------------------------- GSE308395
def clean_GSE308395():
    """South Africa, RNA-seq raw counts, very shallow. Supplementary only."""
    import mygene
    url = f"{FTP}/GSE308nnn/GSE308395/suppl/GSE308395_GB_RNASeq_Raw_Counts.txt.gz"
    c = pd.read_csv(download(url, RAW / "GSE308395_GB_RNASeq_Raw_Counts.txt.gz"),
                    sep="\t", index_col=0)
    c = c[~c.index.astype(str).str.startswith("__")]       # bookkeeping rows, NOT genes
    c.index = c.index.astype(str).str.split(".").str[0]
    c = c[(c >= 1).sum(axis=1) >= 3]

    mg = mygene.MyGeneInfo()
    res = retry(lambda: mg.querymany(c.index.tolist(), scopes="ensembl.gene", fields="symbol",
                                     species="human", verbose=False), label="mygene ensembl")
    mapping = {}
    for r in res:
        if "symbol" in r and r["query"] not in mapping:
            mapping[r["query"]] = r["symbol"]
    c.index = c.index.map(mapping)
    c = c[c.index.notna()].groupby(level=0).sum()
    print(f"   genes with a name: {c.shape[0]}")

    depth = c.sum()                                         # reads that landed on named genes
    depth.rename("gene_reads").to_csv(RES / "GSE308395_depth.csv")
    good = depth >= MIN_GENE_READS                          # rule uses depth only, never labels
    print("   excluded for low depth:", depth[~good].index.tolist())
    c_q = c.loc[:, good]
    c_f = c_q[(c_q >= 5).sum(axis=1) >= 3]
    X = np.log2(c_f / c_f.sum() * 1_000_000 + 1).T
    group = np.where(X.index.str.contains("-GB"), "normal", "tumor")
    labels = pd.DataFrame({"group": group, "patient": range(len(X))}, index=X.index)
    check("samples x genes", X.shape, (9, 3863))
    save_study("GSE308395", X, labels)


# ---------------------------------------------------------------- GSE62335
def clean_GSE62335():
    """China, Affymetrix microarray, 5 tumor + 5 adjacent normal. Supplementary only."""
    import mygene
    gse = get_geo("GSE62335")
    expr = gse.pivot_samples("VALUE")                       # already log2
    meta = gse.phenotype_data
    tissue = meta.filter(like="tissue").iloc[:, 0]
    group = tissue.map({"gallbladder cancer": "tumor", "adjacent normal tissue": "normal"})
    if group.isna().any():
        raise SystemExit(f"Unexpected tissue labels: {tissue.unique()}")
    patient = meta["title"].str.extract(r"(\d+)")[0].astype(int)

    gpl = list(gse.gpls.values())[0].table
    g = gpl[["ID", "GB_ACC"]].dropna().copy()
    g["ID"] = g["ID"].astype(str)
    g["acc"] = g["GB_ACC"].astype(str).str.split(".").str[0]
    accs = g["acc"].unique().tolist()
    mg = mygene.MyGeneInfo()
    res = retry(lambda: mg.querymany(accs, scopes="refseq,accession", fields="symbol",
                                     species="human", verbose=False), label="mygene refseq")
    amap = {}
    for r in res:
        if "symbol" in r and r["query"] not in amap:
            amap[r["query"]] = r["symbol"]
    g["symbol"] = g["acc"].map(amap)
    g = g.dropna(subset=["symbol"]).drop_duplicates("ID")

    e = expr.copy()
    e.index = e.index.astype(str)
    e = e.loc[e.index.isin(g["ID"])]
    e.index = e.index.map(g.set_index("ID")["symbol"])
    X = e.groupby(level=0).mean().T                          # no extra log step
    labels = pd.DataFrame({"group": group.reindex(X.index), "patient": patient.reindex(X.index)})
    check("samples x genes", X.shape, (10, 15630))
    save_study("GSE62335", X, labels)


# ---------------------------------------------------------------- GSE255497
def clean_GSE255497():
    """India (AIIMS Delhi), Agilent microarray: tumor / normal / gallstone."""
    gse = get_geo("GSE255497")
    expr = gse.pivot_samples("VALUE")                       # already log2
    meta = gse.phenotype_data
    gpl = list(gse.gpls.values())[0].table
    dis = meta.filter(like="disease").iloc[:, 0]
    titles = meta["title"]

    def label(name, d):
        if name.startswith("GSD"):
            return "gallstone"
        if re.match(r"^GBC\d+N", name):
            return "normal"
        if re.match(r"^GBC\d+T", name):
            # named tumor but GEO calls it Control -> unclear, kept out of the main analysis
            return "tumor" if str(d).lower() == "adenocarcinoma" else "unclear"
        return "unknown"

    info = pd.DataFrame({"title": titles, "group": [label(n, d) for n, d in zip(titles, dis)]})
    info["patient"] = info["title"].str.extract(r"(\d+)")[0].astype(int)
    print("   raw groups:", info["group"].value_counts().to_dict())

    g = gpl.copy()
    g["ID"] = g["ID"].astype(str)
    g = g[g["CONTROL_TYPE"].astype(str).str.upper() == "FALSE"]          # remove chip test spots
    g = g[g["GENE_SYMBOL"].notna() & (g["GENE_SYMBOL"].astype(str).str.strip() != "")]
    sym = g.drop_duplicates("ID").set_index("ID")["GENE_SYMBOL"]
    e = expr.copy()
    e.index = e.index.astype(str)
    e = e.loc[e.index.isin(sym.index)]
    e.index = e.index.map(sym)
    X = e.groupby(level=0).mean().T
    info = info.reindex(X.index)
    X.index = info["title"].values
    info.index = X.index

    # repeated measurements of one sample: average them if the arrays are near-identical
    for a, b in [("GBC18T", "GBC18T-1"), ("GBC31N", "GBC31N-1")]:
        if a in X.index and b in X.index:
            r = np.corrcoef(X.loc[a], X.loc[b])[0, 1]
            print(f"   {a} vs {b}: correlation {r:.3f}", end=" -> ")
            if r >= 0.95:
                X.loc[a] = (X.loc[a] + X.loc[b]) / 2
                X = X.drop(index=b)
                info = info.drop(index=b)
                print("averaged")
            else:
                print("kept separate")

    main = info["group"] != "unclear"
    check("main samples x genes", X[main].shape, (27, 32080))
    X[~main].to_csv(PROC / "unclear_GSE255497.csv")
    save_study("GSE255497", X[main], info.loc[main, ["group", "patient"]])


STUDIES = {"GSE139682": clean_GSE139682, "GSE308395": clean_GSE308395,
           "GSE62335": clean_GSE62335, "GSE255497": clean_GSE255497}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redo studies that are already cleaned")
    ap.add_argument("--only", nargs="+", choices=list(STUDIES), help="run only these studies")
    args = ap.parse_args()
    for name in (args.only or STUDIES):
        done = (PROC / f"clean_{name}.csv").exists() and (PROC / f"labels_{name}.csv").exists()
        if done and not args.force:
            print(f"{name}: already cleaned (use --force to redo)")
            continue
        print(f"\n=== {name} ===")
        STUDIES[name]()
