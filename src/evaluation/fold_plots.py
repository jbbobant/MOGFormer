"""
fold_plots.py — aggregate monitoring figures for the K-fold MOGFormer run.

Overlay every fold (faint) + mean/aggregate on top, rather than one
figure per fold (clutter) or means only (hides CV variance). All figures PNG+SVG at 300 dpi

  F00_fold_training_curves   train/val loss + inner-val macro-F1 vs epoch, per fold + mean
  F01_perclass_F1_box        per-class F1 distribution across folds (+ macro mean line)
  F02_confusion_aggregate    summed + row-normalized confusion over all folds
  F03_fold_macroF1_strip     per-fold macro-F1 with mean +/- 95% CI
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fold_training_curves(histories, fig_dir, name="F00_fold_training_curves"):
    """histories: list of {'train_loss':[...], 'val_loss':[...], 'val_f1':[...]}."""
    panels = [("train_loss", "train loss"), ("val_loss", "inner-val loss"),
              ("val_f1", "inner-val macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, panels):
        maxlen = max((len(h[key]) for h in histories), default=0)
        for h in histories:
            ax.plot(range(1, len(h[key]) + 1), h[key], color="#999999", alpha=0.25, lw=0.8)
        mean_curve = [np.mean([h[key][e] for h in histories if len(h[key]) > e])
                      for e in range(maxlen)]
        ax.plot(range(1, maxlen + 1), mean_curve, color="#C44E52", lw=2.2, label="mean")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3)
    axes[2].legend(loc="lower right")
    fig.suptitle("Per-fold training curves  (grey = each fold, red = mean over folds)")
    _save(fig, fig_dir, name)


def perclass_f1_box(df, class_names, fig_dir, name="F01_perclass_F1_box"):
    """df: per-fold records with columns f1__<class> and macro_f1."""
    data = [df[f"f1__{c}"].dropna().to_numpy() for c in class_names]
    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 5))
    ax.boxplot(data, tick_labels=class_names, showmeans=True)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(rng.normal(i, 0.04, size=len(d)), d, s=10, alpha=0.5, color="#4C72B0")
    macro_mean = float(df["macro_f1"].mean())
    ax.axhline(macro_mean, ls="--", color="#C44E52", label=f"macro-F1 mean {macro_mean:.3f}")
    ax.set_ylabel("per-class F1 (across folds)"); ax.set_ylim(0, 1.02)
    ax.set_title("Per-class F1 distribution across folds"); ax.legend(loc="lower left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, fig_dir, name)


def confusion_aggregate(confusions, class_names, fig_dir, name="F02_confusion_aggregate"):
    """confusions: list of per-fold N x N count matrices."""
    C = np.sum(confusions, axis=0).astype(float)
    row = C / C.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, M, title, is_count in [(axes[0], C, "summed counts", True),
                                   (axes[1], row, "row-normalized (recall)", False)]:
        im = ax.imshow(M, cmap="Blues"); fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                v = M[i, j]
                ax.text(j, i, f"{int(v)}" if is_count else f"{v:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > M.max() * 0.6 else "black")
    fig.suptitle("Aggregated confusion over all folds")
    _save(fig, fig_dir, name)


def fold_macroF1_strip(df, fig_dir, name="F03_fold_macroF1_strip"):
    from scipy import stats
    v = df["macro_f1"].to_numpy()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.scatter(v, rng.normal(0, 0.02, len(v)), s=26, alpha=0.6, color="#4C72B0")
    mean = float(v.mean())
    ax.axvline(mean, color="#C44E52", lw=2, label=f"mean {mean:.3f}")
    if len(v) > 1:
        sem = v.std(ddof=1) / np.sqrt(len(v))
        h = stats.t.ppf(0.975, len(v) - 1) * sem
        ax.axvspan(mean - h, mean + h, color="#C44E52", alpha=0.15,
                   label=f"95% CI [{mean-h:.3f}, {mean+h:.3f}]")
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("per-fold macro-F1")
    ax.set_title(f"Fold stability (n={len(v)})"); ax.legend(loc="lower left")
    _save(fig, fig_dir, name)


def save_monitoring_figures(df, histories, confusions, class_names, fig_dir):
    """Called once after the CV run from runner._write_baseline_artifacts."""
    if histories:
        fold_training_curves(histories, fig_dir)
    perclass_f1_box(df, class_names, fig_dir)
    if confusions:
        confusion_aggregate(confusions, class_names, fig_dir)
    fold_macroF1_strip(df, fig_dir)
    print(f"  [figures] wrote F00-F03 monitoring figures to {fig_dir}/")