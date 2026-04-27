"""
============================================================
  COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA) SCRIPT
  Usage: python eda.py --file <path_to_csv_or_excel>
         python eda.py --file data.csv --target loan_status
============================================================
"""

import argparse
import warnings
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Aesthetic config ──────────────────────────────────────
PALETTE   = "viridis"
BG_COLOR  = "#0f1117"
FG_COLOR  = "#e8eaf0"
ACC_COLOR = "#4fc3f7"
WARN_COLOR= "#ff7043"
GOOD_COLOR= "#66bb6a"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#333a50",
    "axes.labelcolor":  FG_COLOR,
    "xtick.color":      FG_COLOR,
    "ytick.color":      FG_COLOR,
    "text.color":       FG_COLOR,
    "grid.color":       "#1e2535",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1a2035",
    "legend.edgecolor": "#333a50",
    "font.family":      "DejaVu Sans",
    "figure.dpi":       120,
})

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def banner(text):
    line = "═" * 60
    print(f"\n{line}\n  {text}\n{line}")


def load_data(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        sys.exit(f"Unsupported file type: {ext}")
    print(f"  ✔  Loaded  {df.shape[0]:,} rows × {df.shape[1]} columns  from  '{filepath}'")
    return df


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  💾  Saved → {path}")


def split_columns(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


# ═══════════════════════════════════════════════════════════
#  1.  OVERVIEW
# ═══════════════════════════════════════════════════════════

def section_overview(df):
    banner("1. DATASET OVERVIEW")
    print(df.dtypes.to_string())
    print(f"\nShape : {df.shape}")
    print(f"Duplicate rows : {df.duplicated().sum():,}")
    print("\nMemory usage:")
    print(df.memory_usage(deep=True).to_string())


# ═══════════════════════════════════════════════════════════
#  2.  MISSING VALUES
# ═══════════════════════════════════════════════════════════

def section_missing(df):
    banner("2. MISSING / NaN VALUES")

    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Missing": miss, "Pct (%)": miss_pct})
    miss_df = miss_df[miss_df["Missing"] > 0].sort_values("Pct (%)", ascending=False)

    if miss_df.empty:
        print("  ✅  No missing values found!")
        return

    print(miss_df.to_string())

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, max(4, len(miss_df) * 0.45)))
    colors = [WARN_COLOR if p > 20 else ACC_COLOR for p in miss_df["Pct (%)"]]
    bars = ax.barh(miss_df.index, miss_df["Pct (%)"], color=colors, edgecolor="#222")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Value Analysis", fontsize=14, fontweight="bold", color=FG_COLOR)
    ax.axvline(20, color=WARN_COLOR, linewidth=1.2, linestyle="--", label=">20% threshold")
    for bar, val in zip(bars, miss_df["Pct (%)"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=8, color=FG_COLOR)
    ax.legend()
    plt.tight_layout()
    save(fig, "2_missing_values")

    # Heatmap of missingness pattern (first 500 rows)
    sample = df.isnull().iloc[:500]
    if sample.any().any():
        fig, ax = plt.subplots(figsize=(min(20, len(df.columns) * 0.6), 6))
        sns.heatmap(sample.T, cbar=False, cmap="rocket_r", yticklabels=True, ax=ax)
        ax.set_title("Missingness Pattern (first 500 rows)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Row index")
        plt.tight_layout()
        save(fig, "2_missing_pattern_heatmap")


# ═══════════════════════════════════════════════════════════
#  3.  DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════

def section_describe(df, num_cols):
    banner("3. DESCRIPTIVE STATISTICS (Numeric)")
    desc = df[num_cols].describe().T
    desc["skewness"] = df[num_cols].skew().round(3)
    desc["kurtosis"] = df[num_cols].kurt().round(3)
    print(desc.to_string())


# ═══════════════════════════════════════════════════════════
#  4.  DISTRIBUTIONS  (histograms + KDE)
# ═══════════════════════════════════════════════════════════

def section_distributions(df, num_cols):
    if not num_cols:
        return
    banner("4. NUMERIC DISTRIBUTIONS")

    cols_per_row = 3
    n_rows = -(-len(num_cols) // cols_per_row)          # ceiling division
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(cols_per_row * 5, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=40, color=ACC_COLOR, edgecolor="#222", alpha=0.7, density=True)
        data.plot.kde(ax=ax, color=WARN_COLOR, linewidth=2)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        skew = data.skew()
        ax.text(0.97, 0.95, f"skew={skew:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                color=WARN_COLOR if abs(skew) > 1 else GOOD_COLOR)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "4_distributions")


# ═══════════════════════════════════════════════════════════
#  5.  OUTLIER ANALYSIS  (boxplots + IQR summary)
# ═══════════════════════════════════════════════════════════

def section_outliers(df, num_cols):
    if not num_cols:
        return
    banner("5. OUTLIER ANALYSIS")

    # IQR-based count
    outlier_summary = {}
    for col in num_cols:
        s = df[col].dropna()
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum()
        outlier_summary[col] = n_out

    out_df = pd.Series(outlier_summary).sort_values(ascending=False)
    print("IQR Outlier Counts:")
    print(out_df.to_string())

    # Boxplots
    cols_per_row = 3
    n_rows = -(-len(num_cols) // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(cols_per_row * 5, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        boxprops=dict(facecolor="#1e2f4a", color=ACC_COLOR),
                        medianprops=dict(color=WARN_COLOR, linewidth=2),
                        whiskerprops=dict(color=FG_COLOR),
                        capprops=dict(color=FG_COLOR),
                        flierprops=dict(marker="o", color=WARN_COLOR,
                                        markersize=3, alpha=0.5))
        ax.set_title(col, fontsize=10, fontweight="bold")
        n = outlier_summary[col]
        ax.text(0.97, 0.95, f"{n} outliers", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                color=WARN_COLOR if n > 0 else GOOD_COLOR)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Outlier Boxplots (IQR method)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "5_outlier_boxplots")

    # Bar chart of outlier counts
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors = [WARN_COLOR if v > 0 else GOOD_COLOR for v in out_df.values]
    ax2.bar(out_df.index, out_df.values, color=colors, edgecolor="#222")
    ax2.set_title("IQR Outlier Count per Feature", fontsize=13, fontweight="bold")
    ax2.set_ylabel("# Outliers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save(fig2, "5_outlier_counts")


# ═══════════════════════════════════════════════════════════
#  6.  Z-SCORE OUTLIER HEATMAP
# ═══════════════════════════════════════════════════════════

def section_zscore(df, num_cols):
    if not num_cols:
        return
    banner("6. Z-SCORE ANALYSIS")

    z_df = df[num_cols].apply(lambda x: np.abs(stats.zscore(x.dropna()
              .reindex(x.index).fillna(x.median()))))
    extreme = (z_df > 3).sum().sort_values(ascending=False)
    print("Features with |Z| > 3:")
    print(extreme[extreme > 0].to_string())

    fig, ax = plt.subplots(figsize=(max(10, len(num_cols) * 0.7), 5))
    extreme.plot(kind="bar", ax=ax, color=[
        WARN_COLOR if v > 0 else GOOD_COLOR for v in extreme.values],
        edgecolor="#222")
    ax.set_title("|Z-score| > 3 Count per Feature", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save(fig, "6_zscore_outliers")


# ═══════════════════════════════════════════════════════════
#  7.  CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════

def section_correlation(df, num_cols):
    if len(num_cols) < 2:
        return
    banner("7. CORRELATION MATRIX")

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.8),
                                    max(6, len(num_cols) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=len(num_cols) <= 20,
                fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.3, linecolor="#111",
                annot_kws={"size": 7}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix (lower triangle)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save(fig, "7_correlation_matrix")

    # Top correlated pairs
    corr_pairs = (corr.where(~mask)
                  .stack()
                  .reset_index()
                  .rename(columns={0: "corr", "level_0": "f1", "level_1": "f2"}))
    corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
    top = corr_pairs.nlargest(10, "abs_corr")
    print("\nTop 10 correlated pairs:")
    print(top[["f1", "f2", "corr"]].to_string(index=False))


# ═══════════════════════════════════════════════════════════
#  8.  CATEGORICAL ANALYSIS
# ═══════════════════════════════════════════════════════════

def section_categorical(df, cat_cols):
    if not cat_cols:
        return
    banner("8. CATEGORICAL FEATURE ANALYSIS")

    # Only plot top-N cardinality cols that are meaningful
    valid = [c for c in cat_cols if 1 < df[c].nunique() <= 50]
    if not valid:
        print("  No low-cardinality categorical columns found.")
        return

    cols_per_row = 2
    n_rows = -(-len(valid) // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(cols_per_row * 7, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid):
        ax = axes[i]
        vc = df[col].value_counts().head(20)
        colors = sns.color_palette(PALETTE, len(vc))
        ax.barh(vc.index.astype(str), vc.values, color=colors, edgecolor="#222")
        ax.set_title(f"{col}  (n={df[col].nunique()} unique)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Count")
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Value Counts", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "8_categorical_counts")


# ═══════════════════════════════════════════════════════════
#  9.  TARGET ANALYSIS  (if provided)
# ═══════════════════════════════════════════════════════════

def section_target(df, target, num_cols, cat_cols):
    if target not in df.columns:
        print(f"\n  ⚠  Target column '{target}' not found. Skipping target analysis.")
        return
    banner(f"9. TARGET ANALYSIS  →  '{target}'")

    # Target distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    if df[target].dtype == object or df[target].nunique() <= 20:
        vc = df[target].value_counts()
        colors = [WARN_COLOR, ACC_COLOR, GOOD_COLOR, "#ab47bc", "#ffa726"][:len(vc)]
        ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="#222")
        ax.set_title(f"Target Distribution: {target}", fontsize=13, fontweight="bold")
        for xi, (label, val) in enumerate(vc.items()):
            ax.text(xi, val + len(df) * 0.005, f"{val:,}\n({val/len(df)*100:.1f}%)",
                    ha="center", fontsize=9)
        # class imbalance warning
        imbalance_ratio = vc.max() / vc.min()
        if imbalance_ratio > 3:
            ax.text(0.98, 0.95,
                    f"⚠ Imbalance ratio: {imbalance_ratio:.1f}x",
                    transform=ax.transAxes, ha="right", va="top",
                    color=WARN_COLOR, fontsize=10)
    else:
        ax.hist(df[target].dropna(), bins=40, color=ACC_COLOR, edgecolor="#222", alpha=0.8)
        ax.set_title(f"Target Distribution: {target}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "9_target_distribution")

    # Numeric features vs target (violin)
    is_binary = df[target].nunique() <= 5
    if is_binary and num_cols:
        show = num_cols[:min(9, len(num_cols))]
        cols_per_row = 3
        n_rows = -(-len(show) // cols_per_row)
        fig, axes = plt.subplots(n_rows, cols_per_row,
                                 figsize=(cols_per_row * 5, n_rows * 4))
        axes = np.array(axes).flatten()
        for i, col in enumerate(show):
            ax = axes[i]
            sns.violinplot(data=df, x=target, y=col, ax=ax,
                           palette=PALETTE, inner="box",
                           linewidth=0.8)
            ax.set_title(f"{col} vs {target}", fontsize=9, fontweight="bold")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Feature Distributions by Target Class", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save(fig, "9_numeric_vs_target")


# ═══════════════════════════════════════════════════════════
#  10. PAIRPLOT  (top correlated features)
# ═══════════════════════════════════════════════════════════

def section_pairplot(df, num_cols, target=None):
    if len(num_cols) < 2:
        return
    banner("10. PAIRPLOT (top features by variance)")

    # Pick top 5 by variance to keep it readable
    top_var = df[num_cols].var().nlargest(5).index.tolist()
    plot_df = df[top_var].copy()

    hue = None
    if target and target in df.columns and df[target].nunique() <= 10:
        plot_df[target] = df[target].astype(str)
        hue = target

    g = sns.pairplot(plot_df, hue=hue, diag_kind="kde",
                     plot_kws=dict(alpha=0.4, s=10, edgecolors="none"),
                     diag_kws=dict(fill=True),
                     palette=PALETTE)
    g.figure.suptitle("Pairplot – Top 5 Features by Variance",
                       y=1.02, fontsize=14, fontweight="bold", color=FG_COLOR)
    g.figure.patch.set_facecolor(BG_COLOR)
    for ax in g.axes.flatten():
        ax.set_facecolor(BG_COLOR)
    save(g.figure, "10_pairplot")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Comprehensive EDA Script")
    parser.add_argument("--file",   required=True, help="Path to CSV / Excel / Parquet")
    parser.add_argument("--target", default=None,  help="Target column name (optional)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't call plt.show(); just save files")
    args = parser.parse_args()

    df = load_data(args.file)
    num_cols, cat_cols = split_columns(df)
    print(f"\n  Numeric  columns : {len(num_cols)}")
    print(f"  Categorical cols : {len(cat_cols)}")

    section_overview(df)
    section_missing(df)
    section_describe(df, num_cols)
    section_distributions(df, num_cols)
    section_outliers(df, num_cols)
    section_zscore(df, num_cols)
    section_correlation(df, num_cols)
    section_categorical(df, cat_cols)

    if args.target:
        section_target(df, args.target, num_cols, cat_cols)

    section_pairplot(df, num_cols, target=args.target)

    banner("EDA COMPLETE")
    print(f"  All plots saved to  → ./{OUTPUT_DIR}/\n")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()