"""Figures 1 to 7."""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline

from common import (FIG, LOGREG_C, N_PERM, PANEL_SIZE, PROC, RES, SUPPLEMENTARY, TEST, TRAIN,
                    auc_ci, check, load_study, zscore)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"tumor": "#c0392b", "normal": "#2c6fbb", "gallstone": "#2e8b57", "unclear": "#999999"}
PANEL_COL = ["#2c6fbb", "#c0392b", "#e08a1e"]


def short(name):
    return name.replace(" (= original top-20)", "")


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved figures/{name}")


def load_z():
    Xtr = pd.read_csv(PROC / f"z_{TRAIN}.csv", index_col=0)
    ltr = pd.read_csv(PROC / f"labels_{TRAIN}.csv", index_col=0).reindex(Xtr.index)
    Xin = pd.read_csv(PROC / f"z_{TEST}.csv", index_col=0)[Xtr.columns]
    lin = pd.read_csv(PROC / f"labels_{TEST}.csv", index_col=0).reindex(Xin.index)
    return Xtr, ltr, Xin, lin


def load_panels():
    p = pd.read_csv(RES / "panels.csv")
    return {name: g["gene"].tolist() for name, g in p.groupby("panel", sort=False)}


# ------------------------------------------------------------------ 1. workflow
def fig1():
    core = len(pd.read_csv(PROC / "core_genes.csv"))
    de = pd.read_csv(RES / f"de_{TRAIN}.csv")
    n_sig = int((de["fdr"] < 0.05).sum())
    ind = pd.read_csv(RES / "results_table_india.csv")
    lo, hi = ind["AUC"].min(), ind["AUC"].max()
    steps = [
        ("1  Data search", "4 public GEO studies kept\n(cell-line and drug\nstudies rejected)", "#d9ecec"),
        ("2  Clean", "gene names, log scale,\nQC, repeats merged", "#d9ecec"),
        ("3  Common genes", f"{core:,} genes shared by\nthe China and India\nstudies", "#d9ecec"),
        ("4  Differential\nexpression", f"{n_sig:,} genes differ\n(paired test, FDR < 0.05)", "#d9ecec"),
        ("5  Pathways", "cell cycle up in tumor;\nmuscle and matrix down", "#d9ecec"),
        ("6  Model", "20-gene logistic\nregression, leave-one-\npatient-out CV", "#e6dcf0"),
        ("7  External test", f"India cohort scored once\nAUC {lo:.2f} to {hi:.2f}", "#f8e0c8"),
        ("8  Literature", "40 genes checked in\nPubMed and full texts", "#d9ecec"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    w, h = 0.86, 0.78
    pos = []
    for i, (title, body, colr) in enumerate(steps):
        r, c = divmod(i, 4)
        x, y = c + 0.5, 1.5 - r
        pos.append((x, y))
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=colr, ec="#444", lw=0.8))
        ax.text(x, y + 0.2, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.12, body, ha="center", va="center", fontsize=8)
    arrow = dict(arrowstyle="-|>", color="#444", lw=1.1)
    for i in range(7):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if (i + 1) % 4:
            ax.annotate("", xy=(x2 - w / 2 - 0.02, y2), xytext=(x1 + w / 2 + 0.02, y1), arrowprops=arrow)
        else:
            ax.plot([x1, x1], [y1 - h / 2 - 0.02, 1.0], color="#444", lw=1.1)
            ax.plot([x1, x2], [1.0, 1.0], color="#444", lw=1.1)
            ax.annotate("", xy=(x2, y2 + h / 2 + 0.02), xytext=(x2, 1.0), arrowprops=arrow)
    ax.set_title("Gallbladder cancer biomarker pipeline", fontsize=12, fontweight="bold")
    ax.legend(handles=[Patch(fc="#d9ecec", ec="#444", label="data and biology"),
                       Patch(fc="#e6dcf0", ec="#444", label="machine learning"),
                       Patch(fc="#f8e0c8", ec="#444", label="validation")],
              loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig1_workflow.png")


# ------------------------------------------------------------------ 2. PCA per study
def fig2():
    names = [TRAIN, TEST] + SUPPLEMENTARY
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.ravel(), names):
        X, lab = load_study(name)
        pcs = PCA(2).fit(zscore(X))
        xy = pcs.transform(zscore(X))
        for g in ["normal", "gallstone", "tumor"]:
            m = (lab["group"] == g).values
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], c=COL[g], s=38, label=f"{g} ({m.sum()})",
                           edgecolor="white", lw=0.5)
        ev = pcs.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
        ax.set_title(f"{name}  ({X.shape[0]} samples, {X.shape[1]:,} genes)", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("PCA of each cleaned study (each gene standardized within its study)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig2_pca_studies.png")


# ------------------------------------------------------------------ 3. volcano with panel genes
def fig3():
    de = pd.read_csv(RES / f"de_{TRAIN}.csv")
    panels = load_panels()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sig = de["fdr"] < 0.05
    ax.scatter(de["log2fc"], -np.log10(de["p"]), s=3, c=np.where(sig, "#e8b4ae", "#cfcfcf"), rasterized=True)
    for (name, genes), colr in zip(panels.items(), PANEL_COL):
        sub = de[de["gene"].isin(genes)]
        ax.scatter(sub["log2fc"], -np.log10(sub["p"]), s=22, c=colr, edgecolor="black", lw=0.4,
                   label=f"{short(name)} ({len(sub)})", zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["gene"], (r["log2fc"], -np.log10(r["p"])), fontsize=5.5,
                        xytext=(2, 2), textcoords="offset points")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("log2 fold change (tumor - normal), GSE139682")
    ax.set_ylabel("-log10 p (paired test)")
    ax.set_title("Differential expression, training study only", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    save(fig, "fig3_volcano_panels.png")


# ------------------------------------------------------------------ 4. pathways
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ok = False
    for ax, d, colr in zip(axes, ["UP", "DOWN"], ["#c0392b", "#2c6fbb"]):
        f = RES / f"pathways_{d}_{TRAIN}.csv"
        if not f.exists():
            ax.axis("off")
            ax.set_title(f"{d}: run step 4 first")
            continue
        t = pd.read_csv(f)
        t["label"] = t["Term"].map(lambda s: re.sub(r"\s*(\(GO:\d+\)|R-HSA-\d+)\s*$", "", str(s)))
        t = t.sort_values("Adjusted P-value").drop_duplicates("label").head(8).iloc[::-1]
        ax.barh([x if len(x) < 48 else x[:45] + "..." for x in t["label"]],
                -np.log10(t["Adjusted P-value"]), color=colr)
        ax.set_xlabel("-log10 adjusted p")
        ax.set_title(f"{'Higher' if d == 'UP' else 'Lower'} in tumor", fontsize=10)
        ok = True
    if not ok:
        plt.close(fig)
        raise FileNotFoundError("pathway files missing")
    fig.suptitle("Top enriched pathways (GO, Reactome, KEGG; background = tested genes)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_pathways.png")


# ------------------------------------------------------------------ 5. CV + shuffled labels
def perm_null(Xtr, ytr, grp, skip):
    f = RES / "permutation_null.csv"
    if f.exists():
        return pd.read_csv(f)["auc"].values, None
    if skip:
        return None, None

    def cv_auc(y):
        pipe = Pipeline([("sel", SelectKBest(f_classif, k=PANEL_SIZE)),
                         ("clf", LogisticRegression(C=LOGREG_C, max_iter=5000))])
        s = cross_val_predict(pipe, Xtr.values, y, groups=grp, cv=LeaveOneGroupOut(),
                              method="decision_function")
        return roc_auc_score(y, s)

    rng = np.random.default_rng(0)                       # same recipe as step 5
    real = cv_auc(ytr)
    null = []
    for _ in range(N_PERM):
        yp = ytr.copy()
        for pid in np.unique(grp):
            if rng.random() < 0.5:
                idx = np.where(grp == pid)[0]
                yp[idx] = 1 - yp[idx]
        null.append(cv_auc(yp))
    pd.DataFrame({"auc": null}).to_csv(f, index=False)
    return np.array(null), real


def fig5(skip_perm):
    Xtr, ltr, _, _ = load_z()
    ytr = (ltr["group"] == "tumor").astype(int).values
    grp = ltr["patient"].values
    cv = pd.read_csv(RES / "cv_table.csv", index_col=0)
    null, real = perm_null(Xtr, ytr, grp, skip_perm)
    fig, axes = plt.subplots(1, 2 if null is not None else 1, figsize=(10.5 if null is not None else 5.5, 4.2))
    axes = np.atleast_1d(axes)
    for c, colr, lbl in [("logreg", "#2c6fbb", "logistic regression"), ("forest", "#e08a1e", "random forest")]:
        axes[0].plot(cv.index, cv[c], "o-", color=colr, label=lbl)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_xlabel("genes kept inside each fold")
    axes[0].set_ylabel("AUC (leave-one-patient-out)")
    axes[0].axvline(PANEL_SIZE, color="grey", ls=":", lw=1)
    axes[0].set_title("Training study: cross-validation", fontsize=10)
    axes[0].legend(frameon=False)
    if null is not None:
        if real is None:
            real = cv.loc[PANEL_SIZE, "logreg"]
        axes[1].hist(null, bins=20, color="#bbbbbb", edgecolor="white")
        axes[1].axvline(real, color="#c0392b", lw=2, label=f"real labels: AUC {real:.2f}")
        p = (1 + (null >= real - 1e-9).sum()) / (1 + len(null))
        axes[1].set_xlabel("AUC with tumor/normal labels swapped within patients")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"{len(null)} shuffles, p = {p:.3f}", fontsize=10)
        axes[1].legend(frameon=False)
        print(f"   shuffled mean {null.mean():.3f}, highest {null.max():.3f}, p = {p:.4f}")
    else:
        print("   shuffled-label panel skipped (--skip-perm and no saved results/permutation_null.csv)")
    fig.tight_layout()
    save(fig, "fig5_validation.png")


# ------------------------------------------------------------------ 6. heatmap
def fig6():
    Xtr, ltr, Xin, lin = load_z()
    panels = load_panels()
    genes = [g for gs in panels.values() for g in gs]
    order = {"normal": 0, "gallstone": 1, "tumor": 2}

    def block(X, lab):
        idx = sorted(X.index, key=lambda s: (order.get(lab.loc[s, "group"], 9), s))
        return X.loc[idx, genes].T, lab.loc[idx, "group"].values

    A, ga = block(Xtr, ltr)
    B, gb = block(Xin, lin)
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, height_ratios=[0.25, 10], width_ratios=[A.shape[1], B.shape[1]], hspace=0.02, wspace=0.08)
    for k, (M, g, title) in enumerate([(A, ga, f"{TRAIN} (training, China)"), (B, gb, f"{TEST} (external test, India)")]):
        bar = fig.add_subplot(gs[0, k])
        bar.imshow(np.array([[matplotlib.colors.to_rgb(COL[x]) for x in g]]), aspect="auto")
        bar.axis("off")
        bar.set_title(title, fontsize=10)
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.set_xticks([])
        if k == 0:
            ax.set_yticks(range(len(genes)))
            ax.set_yticklabels(genes, fontsize=6.5)
            start = 0
            for name, gl in panels.items():
                ax.axhline(start + len(gl) - 0.5, color="black", lw=1.2)
                start += len(gl)
        else:
            ax.set_yticks([])
    cax = fig.add_axes([0.93, 0.35, 0.012, 0.3])
    fig.colorbar(im, cax=cax, label="z-score (within study)")
    fig.legend(handles=[Patch(fc=COL[x], label=x) for x in ["normal", "gallstone", "tumor"]],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"The {len(genes)} panel genes (top block: " + short(list(panels)[0]) + ")", fontsize=11)
    save(fig, "fig6_heatmap.png")


# ------------------------------------------------------------------ 7. India scores
def fig7():
    Xtr, ltr, Xin, lin = load_z()
    ytr = (ltr["group"] == "tumor").astype(int).values
    panels = load_panels()
    table = pd.read_csv(RES / "results_table_india.csv")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(1)
    for ax, (name, genes) in zip(axes, panels.items()):
        m = LogisticRegression(C=LOGREG_C, max_iter=5000).fit(Xtr[genes].values, ytr)
        s = pd.Series(m.decision_function(Xin[genes].values), index=Xin.index)
        data = [s[lin["group"] == g].values for g in ["normal", "gallstone", "tumor"]]
        ax.boxplot(data, positions=[0, 1, 2], widths=0.5, showfliers=False,
                   medianprops=dict(color="black"), boxprops=dict(color="#555"))
        for i, g in enumerate(["normal", "gallstone", "tumor"]):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(data[i])), data[i], s=26, c=COL[g],
                       edgecolor="white", lw=0.5, zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"normal\n(n={len(data[0])})", f"gallstone\n(n={len(data[1])})",
                            f"tumor\n(n={len(data[2])})"])
        y = (lin["group"] == "tumor").astype(int).values
        a, ci = auc_ci(y, s.values)
        ref = table[(table["panel"] == name) & (table["comparison"] == "Tumor vs all controls")]
        if len(ref):
            check(f"{short(name)} AUC vs saved table", round(a, 2), float(ref["AUC"].iloc[0]))
        ax.set_title(f"{short(name)}\ntumor vs all controls: AUC {a:.2f} ({ci[0]:.2f} to {ci[1]:.2f})", fontsize=9)
        ax.set_ylabel("model score (higher = more tumor-like)")
    fig.suptitle(f"External test on {TEST} (India): each sample scored once", fontsize=11)
    fig.tight_layout()
    save(fig, "fig7_india_scores.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-perm", action="store_true")
    ap.add_argument("--only", nargs="+", type=int)
    args = ap.parse_args()
    jobs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: lambda: fig5(args.skip_perm), 6: fig6, 7: fig7}
    for n, fn in jobs.items():
        if args.only and n not in args.only:
            continue
        print(f"figure {n}")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"   FAILED: {type(e).__name__}: {e}")
