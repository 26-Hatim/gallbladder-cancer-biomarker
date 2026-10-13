"""Pathway enrichment of the changed genes (Enrichr)."""
import time

import pandas as pd

from common import PROC, RES, TRAIN

try:
    import gseapy as gp
except ImportError:
    raise SystemExit("pip install gseapy")

res = pd.read_csv(RES / f"de_{TRAIN}.csv")
background = pd.read_csv(PROC / "core_genes.csv")["gene"].tolist()
up = res[(res.fdr < 0.05) & (res.log2fc > 0)]["gene"].tolist()
down = res[(res.fdr < 0.05) & (res.log2fc < 0)]["gene"].tolist()
print("up in tumor:", len(up), "| down in tumor:", len(down))

LIBS = ["GO_Biological_Process_2023", "Reactome_2022", "KEGG_2021_Human"]


def run(genes, lib, use_bg, tries=3):
    for i in range(tries):
        try:
            kw = dict(gene_list=genes, gene_sets=lib, outdir=None)
            if use_bg:
                kw["background"] = background
            return gp.enrichr(**kw).results
        except Exception as e:  # noqa: BLE001
            print(f"   retry {i + 1} | {lib} | bg={use_bg} | {type(e).__name__}")
            time.sleep(5)
    return None


for name, genes in [("UP", up), ("DOWN", down)]:
    frames = []
    for lib in LIBS:
        r = run(genes, lib, True)
        note = "custom background"
        if r is None:
            r = run(genes, lib, False)
            note = "NO custom background (p-values look stronger than they are)"
        if r is None:
            print(f"   {lib}: FAILED, skipped")
            continue
        r["background_note"] = note
        frames.append(r)
    if not frames:
        print(name, ": nothing worked. Try again later.")
        continue
    out = pd.concat(frames)
    out = out[out["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
    out.to_csv(RES / f"pathways_{name}_{TRAIN}.csv", index=False)
    print(f"\n===== {name} in tumor: {len(out)} significant terms =====")
    print(out[["Gene_set", "Term", "Adjusted P-value", "background_note"]].head(10).to_string(index=False))
print("\nReminder: muscle/matrix pathways lower in tumor may partly reflect tissue composition.")
