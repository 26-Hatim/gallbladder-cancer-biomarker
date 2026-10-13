"""Shared settings and helpers."""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------ folders
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"            # downloaded files (cached, never re-downloaded)
PROC = ROOT / "data" / "processed"     # cleaned tables
RES = ROOT / "results"                 # tables and text results
FIG = ROOT / "figures"                 # pictures
for _d in (RAW, PROC, RES, FIG):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------- decisions fixed in advance
TRAIN = "GSE139682"                    # China, RNA-seq, 10 tumor / 10 matched normal
TEST = "GSE255497"                     # India, microarray (the honest external test)
SUPPLEMENTARY = ["GSE62335", "GSE308395"]
PANEL_SIZE = 20                        # genes in each panel
LOGREG_C = 0.1
MIN_PANEL_GENES = 15                   # a panel is scored on a study only if >= 15 of 20 genes exist
MIN_GENE_READS = 10_000                # RNA-seq samples with fewer gene reads are excluded
N_PERM = 200
N_BOOT = 2000
SEED = 42


# ------------------------------------------------------------------ helpers
def retry(fn, tries=3, wait=5, label="step"):
    """Run fn(); if it fails (network hiccup), wait and try again."""
    last = None
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"   [{label}] attempt {i}/{tries} failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after {tries} attempts") from last


def download(url, dest):
    """Download once and cache. Tells you what to do by hand if it fails."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    def _get():
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        if tmp.stat().st_size < 1000:
            raise IOError("downloaded file is too small")
        tmp.replace(dest)

    try:
        retry(_get, label=f"download {dest.name}")
    except RuntimeError:
        raise SystemExit(
            f"\nCould not download:\n  {url}\n"
            f"Download it by hand from the GEO page and save it exactly as:\n  {dest}\n"
            "then run this script again."
        )
    return dest


def _gzip_ok(path):
    """True if a .gz file can be read to the end (i.e. the download is complete)."""
    import gzip
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:  # noqa: BLE001
        return False


def get_geo(accession):
    """Load a GEO series with GEOparse.

    GEOparse's own FTP download often stops early ("size do not match"), so we fetch the
    SOFT file ourselves over HTTPS, check it is complete, and give the file to GEOparse.
    """
    import GEOparse
    logging.getLogger("GEOparse").setLevel(logging.WARNING)
    folder = accession[:-3] + "nnn"                       # GSE62335 -> GSE62nnn
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder}/{accession}/soft/{accession}_family.soft.gz"
    dest = RAW / f"{accession}_family.soft.gz"
    if dest.exists() and not _gzip_ok(dest):              # broken file from an earlier attempt
        print(f"   removing incomplete {dest.name}")
        dest.unlink()
    download(url, dest)
    if not _gzip_ok(dest):
        dest.unlink()
        raise SystemExit(f"{dest.name} downloaded incompletely. Run the script again.")
    return GEOparse.get_GEO(filepath=str(dest), silent=True)


def zscore(df):
    """Standardize each gene (column) inside one study."""
    return (df - df.mean()) / df.std(ddof=0).replace(0, 1)


def save_study(name, X, labels):
    X.to_csv(PROC / f"clean_{name}.csv")
    labels.to_csv(PROC / f"labels_{name}.csv")
    print(f"   saved clean_{name}.csv {X.shape} and labels_{name}.csv")


def load_study(name):
    X = pd.read_csv(PROC / f"clean_{name}.csv", index_col=0)
    labels = pd.read_csv(PROC / f"labels_{name}.csv", index_col=0).reindex(X.index)
    return X, labels


def auc_ci(y, s, n=N_BOOT, seed=1):
    """AUC with a bootstrap 95% confidence interval."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.percentile(vals, [2.5, 97.5])


def check(label, got, expected):
    """Print whether a value matches the expected one."""
    flag = "OK       " if got == expected else "DIFFERENT"
    print(f"   [{flag}] {label}: got {got}  (expected {expected})")


def resolve_symbols(genes):
    """Map each gene name to its current official symbol.

    Exact official symbols are matched first. Only names that are NOT current symbols
    (old names such as PPAP2B) fall back to an alias search. This avoids a trap: HLF is
    an alias of EPAS1 and FHL1 is an alias of CFH, so an alias-first search returns the
    wrong gene for them.
    Returns {gene: {"symbol", "name", "summary", "note"}}.
    """
    import mygene
    mg = mygene.MyGeneInfo()

    def q(scopes, ids):
        return retry(lambda: mg.querymany(ids, scopes=scopes, fields="symbol,name,summary",
                                          species="human", verbose=False), label="mygene " + scopes)

    out = {}
    for r in q("symbol", list(genes)):
        g = r["query"]
        if g not in out and str(r.get("symbol", "")).upper() == g.upper():
            out[g] = {"symbol": r["symbol"], "name": r.get("name", ""),
                      "summary": r.get("summary", ""), "note": "exact symbol"}
    missing = [g for g in genes if g not in out]
    if missing:
        for r in q("alias", missing):
            g = r["query"]
            if g not in out and "symbol" in r:
                out[g] = {"symbol": r["symbol"], "name": r.get("name", ""),
                          "summary": r.get("summary", ""), "note": "OLD NAME -> check by hand"}
    for g in genes:
        out.setdefault(g, {"symbol": "NOT FOUND", "name": "", "summary": "", "note": "NOT FOUND"})
    return out
