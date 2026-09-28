#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_compare_filters_by_acc.py: figure 11 (one panel per target acceptance probability).
    - Each filter -> one base color; within a filter, code distance d -> light to dark shade.
    - 95% Clopper-Pearson intervals from the saved counts are drawn as error bars.
    - A point with zero failures (if any) is drawn as its 95% upper bound with a downward caret.
Usage: python plot_compare_filters_by_acc.py [input_dir]   (default: current directory)
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ACCEPT_LIST = [10, 20, 40, 80]
IN_FILES = {
    "SW + MWPM": "threshold_scan_SyndromeWeight_acc{acc}.csv",
    "DS + MWPM":      "threshold_scan_ZCosetGap_acc{acc}.csv",
    "DS + NN":       "threshold_scan_NNFilter_acc{acc}.csv",
    # "riskNN":       "threshold_scan_NNRisk_acc{acc}.csv",
}
OUT_FILE = "compare_acc{acc}.pdf"

# base color per filter
FILTER_BASE_COLOR = {
    "SW + MWPM": "tab:blue",
    "DS + MWPM":      "tab:red",
    "DS + NN":       "tab:green",
    # "riskNN":       "tab:orange",
}

FILTER_STYLES = {
    "SW + MWPM": dict(linestyle=":", marker="o", linewidth=2.0),
    "DS + MWPM":      dict(linestyle="--", marker="s", linewidth=2.0),
    "DS + NN":       dict(linestyle="-", marker="^", linewidth=2.0),
    # "riskNN":       dict(linestyle="-.", marker="d", linewidth=2.0),
}
Y_COL = "condL"

def load_csv(path, tag):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["filter"] = tag
    return df

def lighten_color(base_color, factor):
    """factor in [0,1]: 0=black, 1=original."""
    c = np.array(mcolors.to_rgb(base_color))
    return np.clip((1 - factor) * np.ones_like(c) + factor * c, 0, 1)

def plot_acc(acc_int):
    dfs = []
    for name, pattern in IN_FILES.items():
        path = pattern.format(acc=acc_int)
        dfs.append(load_csv(path, name))
    df = pd.concat(dfs, ignore_index=True)

    for c in ["d"]: df[c] = df[c].astype(int)
    for c in ["p_local", "p_trans", Y_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[Y_COL])]
    if df.empty:
        print(f"No finite data for acc={acc_int}")
        return

    x_col = "p_trans" if df["p_trans"].nunique() > 1 else "p_local"
    all_d = sorted(df["d"].unique())
    min_d, max_d = min(all_d), max(all_d)

    plt.figure(figsize=(6.5, 4.5))

    for filt_name, style in FILTER_STYLES.items():
        base_color = FILTER_BASE_COLOR[filt_name]
        subdf = df[df["filter"] == filt_name]
        if subdf.empty:
            continue

        for d_val in all_d:
            g = subdf[subdf["d"] == d_val].copy()
            if g.empty:
                continue
            g = g.sort_values(x_col)

            # compute lightness factor (0→darkest, 1→lightest)
            frac = (d_val - min_d) / (max_d - min_d + 1e-9)
            color = lighten_color(base_color, 0.5 + 0.55 * frac)  # lighter for smaller d

            #: points with failures > 0 form the curve; zero-failure points are upper bounds
            gp = g[g["failures"] > 0]
            gz = g[g["failures"] == 0]

            label = f"d={d_val}" if filt_name == "DS + MWPM" else None
            plt.plot(gp[x_col], gp[Y_COL],
                     color=color, label=label, **style)

            #: 95% Clopper-Pearson intervals from the counts
            plt.errorbar(gp[x_col], gp[Y_COL],
                         yerr=[gp[Y_COL] - gp["ci95_lo"], gp["ci95_hi"] - gp[Y_COL]],
                         fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, capthick=1.0)
            if not gz.empty:
                plt.errorbar(gz[x_col], gz["ci95_hi"], yerr=0.4 * gz["ci95_hi"], uplims=True,
                             fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, capthick=1.0)
                plt.plot(gz[x_col], gz["ci95_hi"], linestyle="none", marker=style["marker"],
                         markerfacecolor="white", markeredgecolor=color)

    plt.yscale("log")
    plt.xlabel("Transmission Error Rate", fontweight="bold")
    plt.ylabel("Logical Error Rate", fontweight="bold")
    # plt.title(f"Acceptance ≈ {acc_int}%, $p_\\mathrm{{local}}=0.003$")
    plt.grid(True, which="both", ls=":", alpha=0.5)

    ax = plt.gca()
    # --- Legend 1: code distance ---
    handles, labels = ax.get_legend_handles_labels()
    gray_handles = []
    gray_cmap = plt.cm.Greys
    n = len(handles)
    for i, (h, lab) in enumerate(zip(handles, labels)):
        gray_shade = gray_cmap(0.4 + 0.5 * (i / max(1, n - 1)))  # 0.4–0.9 light→dark
        line = plt.Line2D([0], [0], color=gray_shade,
                        linestyle=h.get_linestyle(),
                        marker=h.get_marker(),
                        linewidth=h.get_linewidth(),
                        label=lab)
        gray_handles.append(line)

    color_legend = ax.legend(gray_handles, labels, title="Code distance",
                            loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax.add_artist(color_legend)

    # --- Legend 2: filter type ---
    style_handles = [
        plt.Line2D([0], [0], color=FILTER_BASE_COLOR[name], **FILTER_STYLES[name], label=name)
        for name in FILTER_STYLES
    ]
    filt_legend = ax.legend(handles=style_handles, title="Filter + Decoder",
              loc="lower right", bbox_to_anchor=(0.72, 0.0))

    plt.tight_layout()
    #: if a legend box covers a data point or an error bar, lower the bottom of the y axis
    # (in steps of 10**0.25) until no point is covered; legends, styles and figure size are unchanged.
    pts = []
    for filt_name in FILTER_STYLES:
        sub = df[df["filter"] == filt_name]
        for _, r in sub.iterrows():
            if r["failures"] > 0:
                pts.append((r[x_col], r["ci95_lo"], r["ci95_hi"]))
            else:
                pts.append((r[x_col], 0.6 * r["ci95_hi"], r["ci95_hi"]))
    fig = plt.gcf()
    for _iter in range(40):
        fig.canvas.draw()
        boxes = [lg.get_window_extent() for lg in (color_legend, filt_legend)]
        m = 4.0
        hit = False
        for (x, lo, hi) in pts:
            (xd, yl), (_xd2, yh) = ax.transData.transform([(x, max(lo, 1e-300)), (x, hi)])
            for b in boxes:
                if b.x0 - m <= xd <= b.x1 + m and yl <= b.y1 + m and yh >= b.y0 - m:
                    hit = True
        if not hit:
            break
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 / 10 ** 0.25, y1)
    out_pdf = OUT_FILE.format(acc=acc_int)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.03)
    print(f"Saved {out_pdf}")

def main():
    for acc in ACCEPT_LIST:
        try:
            plot_acc(acc)
        except Exception as e:
            print(f"[acc={acc}] ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:          #: input/output directory
        os.chdir(sys.argv[1])
    plt.rcParams.update({
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 14,
        "pdf.fonttype": 42,        #
        "ps.fonttype": 42,         #
    })
    main()
