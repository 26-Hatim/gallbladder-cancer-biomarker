"""Paired differential expression on the training study."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

from common import FIG, PROC, RES, TRAIN, check, load_study

X, lab = load_study(TRAIN)
core = pd.read_csv(PROC / "core_genes.csv")["gene"].tolist()
X = X[[g for g in core if g in X.columns]]

tum = X[lab["group"] == "tumor"].copy()
nor = X[lab["group"] == "normal"].copy()
tum.index = lab.loc[tum.index, "patient"].values
nor.index = lab.loc[nor.index, "patient"].values
nor = nor.loc[tum.index]                                   # same patient order

t, p = ttest_rel(tum, nor)
res = pd.DataFrame({"gene": X.columns, "t": t, "p": p,
                    "log2fc": (tum.values - nor.values).mean(axis=0)}).dropna()
res["fdr"] = multipletests(res["p"], method="fdr_bh")[1]
res = res.sort_values("p")
res.to_csv(RES / f"de_{TRAIN}.csv", index=False)

print("genes tested:", len(res))
check("FDR < 0.05", int((res.fdr < 0.05).sum()), 2225)
print("FDR < 0.05 and |log2FC| > 1:", int(((res.fdr < 0.05) & (res.log2fc.abs() > 1)).sum()))
print("p < 0.01 observed:", int((res.p < 0.01).sum()), "| expected by chance:", round(len(res) * 0.01))
print(res.head(15).round(4).to_string(index=False))

sig = res.fdr < 0.05
plt.figure(figsize=(6, 5))
plt.scatter(res.log2fc, -np.log10(res.p), s=4, c=np.where(sig, "red", "grey"))
plt.xlabel("log2 fold change (tumor - normal)")
plt.ylabel("-log10 p")
plt.title(f"{TRAIN} paired test")
plt.savefig(FIG / f"volcano_{TRAIN}.png", dpi=300, bbox_inches="tight")
print("saved results/de_%s.csv and figures/volcano_%s.png" % (TRAIN, TRAIN))
