"""Cross-validation, permutation test, external test and supplementary check."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline

from common import (FIG, LOGREG_C, MIN_PANEL_GENES, N_PERM, PANEL_SIZE, PROC, RES, SEED,
                    SUPPLEMENTARY, TEST, TRAIN, auc_ci, load_study, zscore)

# ----------------------------------------------------------------- data
Xtr = pd.read_csv(PROC / f"z_{TRAIN}.csv", index_col=0)
ltr = pd.read_csv(PROC / f"labels_{TRAIN}.csv", index_col=0).reindex(Xtr.index)
ytr = (ltr["group"] == "tumor").astype(int).values
grp = ltr["patient"].values
Xin = pd.read_csv(PROC / f"z_{TEST}.csv", index_col=0)[Xtr.columns]
gin = pd.read_csv(PROC / f"labels_{TEST}.csv", index_col=0)["group"].reindex(Xin.index)
print(f"training: {Xtr.shape}, tumors {ytr.sum()}, normals {(1 - ytr).sum()}")
print("India groups:", gin.value_counts().to_dict())


def make_pipe(k=PANEL_SIZE, model="logreg"):
    clf = (LogisticRegression(C=LOGREG_C, max_iter=5000) if model == "logreg"
           else RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
    return Pipeline([("sel", SelectKBest(f_classif, k=k)), ("clf", clf)])


# ----------------------------------------------------------------- A. cross-validation
print("\nA. Leave-one-patient-out cross-validation (training study only)")
rows = []
for model in ["logreg", "forest"]:
    for k in [10, 20, 50, 100, 500]:
        meth = "decision_function" if model == "logreg" else "predict_proba"
        s = cross_val_predict(make_pipe(k, model), Xtr.values, ytr, groups=grp,
                              cv=LeaveOneGroupOut(), method=meth)
        s = s[:, 1] if meth == "predict_proba" else s
        rows.append((model, k, round(roc_auc_score(ytr, s), 3)))
cv_table = pd.DataFrame(rows, columns=["model", "genes", "AUC"]).pivot(
    index="genes", columns="model", values="AUC")
cv_table.to_csv(RES / "cv_table.csv")
print(cv_table)
print("Model fixed before the external test: logistic regression, top %d genes." % PANEL_SIZE)

# ----------------------------------------------------------------- B. permutation test
print("\nB. Permutation test (labels swapped within patients)")


def cv_auc(y):
    s = cross_val_predict(make_pipe(), Xtr.values, y, groups=grp, cv=LeaveOneGroupOut(),
                          method="decision_function")
    return roc_auc_score(y, s)


rng = np.random.default_rng(0)
real = cv_auc(ytr)
null = []
for _ in range(N_PERM):
    yp = ytr.copy()
    for pid in np.unique(grp):
        if rng.random() < 0.5:
            idx = np.where(grp == pid)[0]
            yp[idx] = 1 - yp[idx]
    null.append(cv_auc(yp))
null = np.array(null)
pval = (1 + (null >= real).sum()) / (1 + len(null))
print(f"real AUC {real:.3f} | shuffled mean {null.mean():.3f}, highest {null.max():.3f} | p = {pval:.4f}")

# ----------------------------------------------------------------- panels (fixed here)
diff = Xtr[ytr == 1].mean() - Xtr[ytr == 0].mean()          # training study only


def fit_panel(cols):
    m = make_pipe()
    m.fit(Xtr[cols].values, ytr)
    return m, list(pd.Index(cols)[m.named_steps["sel"].get_support()])


m_all, genes_all = fit_panel(Xtr.columns)
m_up, genes_up = fit_panel(Xtr.columns[diff > 0])
m_dn, genes_dn = fit_panel(Xtr.columns[diff < 0])
if set(genes_all) == set(genes_dn):
    print("\nThe original top-20 panel equals the 'lower in tumor' panel (all 20 genes go down).")
    panels = {"Muscle side (= original top-20)": (m_dn, genes_dn, Xtr.columns[diff < 0]),
              "Tumor-high side": (m_up, genes_up, Xtr.columns[diff > 0])}
else:
    panels = {"Original top-20 (all genes)": (m_all, genes_all, Xtr.columns),
              "Tumor-high side": (m_up, genes_up, Xtr.columns[diff > 0]),
              "Tumor-low side": (m_dn, genes_dn, Xtr.columns[diff < 0])}
pd.DataFrame([(g, name) for name, (_, genes, _) in panels.items() for g in genes],
             columns=["gene", "panel"]).to_csv(RES / "panels.csv", index=False)
for name, (_, genes, _) in panels.items():
    print(f"\n{name}:", genes)

# ----------------------------------------------------------------- C. India test
print("\nC. External test on %s (each panel scored once)" % TEST)
tests = {"Tumor vs normal": ["tumor", "normal"],
         "Tumor vs gallstone": ["tumor", "gallstone"],
         "Tumor vs all controls": ["tumor", "normal", "gallstone"]}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
rows = []
for name, (m, genes, cols) in panels.items():
    sc = m.decision_function(Xin[cols].values)
    for ax, (tname, keep) in zip(axes, tests.items()):
        mask = gin.isin(keep).values
        y = (gin[mask] == "tumor").astype(int).values
        a, ci = auc_ci(y, sc[mask])
        fpr, tpr, _ = roc_curve(y, sc[mask])
        ax.plot(fpr, tpr, label=f"{name}: AUC {a:.2f}")
        rows.append({"panel": name, "comparison": tname, "n_tumor": int(y.sum()),
                     "n_control": int((1 - y).sum()), "AUC": round(a, 2),
                     "CI_low": round(ci[0], 2), "CI_high": round(ci[1], 2)})
for ax, tname in zip(axes, tests):
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(tname)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right")
plt.suptitle(f"External test on {TEST} (India)")
plt.tight_layout()
plt.savefig(FIG / "roc_india.png", dpi=300)
india = pd.DataFrame(rows)
india.to_csv(RES / "results_table_india.csv", index=False)
print(india.to_string(index=False))

# ----------------------------------------------------------------- D. supplementary studies
print("\nD. Supplementary studies (rule: need >= %d of %d panel genes)" % (MIN_PANEL_GENES, PANEL_SIZE))
supp_rows = []
for study in SUPPLEMENTARY:
    X, lab = load_study(study)
    y = (lab["group"] == "tumor").astype(int).values
    print(f"\n{study}: {y.sum()} tumor, {(1 - y).sum()} normal")
    for name, (_, genes, _) in panels.items():
        have = [g for g in genes if g in X.columns]
        line = f"   {name}: {len(have)} of {len(genes)} genes present"
        if len(have) < MIN_PANEL_GENES:
            print(line, "-> skipped")
            supp_rows.append({"study": study, "panel": name, "genes_present": len(have), "AUC": "skipped"})
            continue
        clf = LogisticRegression(C=LOGREG_C, max_iter=5000).fit(Xtr[have].values, ytr)
        s = clf.decision_function(zscore(X[have]).values)
        a, ci = auc_ci(y, s)
        print(line, f"-> AUC {a:.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f})")
        supp_rows.append({"study": study, "panel": name, "genes_present": len(have),
                          "AUC": round(a, 2), "CI_low": round(ci[0], 2), "CI_high": round(ci[1], 2)})
pd.DataFrame(supp_rows).to_csv(RES / "results_supplementary.csv", index=False)
print("\nsaved results/*.csv and figures/roc_india.png")
