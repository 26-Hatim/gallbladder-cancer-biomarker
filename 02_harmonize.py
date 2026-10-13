"""Common gene list and per-study standardization."""
import pandas as pd

from common import PROC, SEED, TEST, TRAIN, SUPPLEMENTARY, check, load_study, zscore

data = {name: load_study(name)[0] for name in [TRAIN, TEST] + SUPPLEMENTARY}
for k, v in data.items():
    print(f"{k}: {v.shape}")

names = list(data)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"   shared genes {names[i]} & {names[j]}: "
              f"{len(set(data[names[i]].columns) & set(data[names[j]].columns))}")

core = sorted(set(data[TRAIN].columns) & set(data[TEST].columns))
check("core genes (2 main studies)", len(core), 14379)
pd.Series(core, name="gene").to_csv(PROC / "core_genes.csv", index=False)
for k in (TRAIN, TEST):
    zscore(data[k][core]).to_csv(PROC / f"z_{k}.csv")
print("saved core_genes.csv, z_%s.csv, z_%s.csv" % (TRAIN, TEST))
