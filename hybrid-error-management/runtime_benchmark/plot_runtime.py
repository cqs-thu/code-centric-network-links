#!/usr/bin/env python3
"""
Redraw Fig 12 from runtime_results_regen.csv (tag A), derived from plot_runtime() in
../plot_running_time.py.  Kept from the original: figure size (10 x 4.5 in), base colours
(SW tab:blue, DS+MWPM tab:red, DS+NN tab:green), shading by code distance (frac = 0.5 ... 0.9),
log y axis "Runtime (µs)", x axis "Transmission Error Rate" with one group per p_trans, dotted y grid,
the two legends ("Code distance" in grey shades, "Filter + Decoder") outside on the right, rcParams.
Changed (the original overlaid the three filters at the same x position, so only the tallest bar was
fully visible and the caption described the encoding wrongly):
  * grouped bars, nothing overlaid or stacked: within each p_trans group one cluster per filter-decoder
    pair (fixed order SW+MWPM | DS+MWPM | DS+NN on GPU | DS+NN on one CPU core), and within each
    cluster one bar per code distance d = 5, 7, 9, 11 (left to right, light to dark);
  * DS+NN on one CPU core is a separate, hatched cluster (same hue as DS+NN, the method is the same);
  * error bars: +- one sample standard deviation over the repetitions;
  * the device is named in the filter legend; pdf.fonttype = ps.fonttype = 42 (no Type-3 fonts).
Every plotted number is t_round_us_mean / t_round_us_std copied from the CSV written by summarize.py.

Usage: python3 plot_runtime.py [--tag A] [--csv runtime_results_regen.csv]
Writes figs/runtime.pdf (500-round batches) and figs/runtime_latency.pdf (batch size 1), + PNG previews.
"""
import os
import sys
import argparse

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

FILTER_BASE_COLOR = {
    "SW + MWPM": "tab:blue",
    "DS + MWPM": "tab:red",
    "DS + NN": "tab:green",
}
# (method, device) clusters in fixed order
SERIES = [("SW + MWPM", "cpu1"), ("DS + MWPM", "cpu1"), ("DS + NN", "gpu"), ("DS + NN", "cpu1")]
SERIES_LABEL = {
    ("SW + MWPM", "cpu1"): "SW + MWPM (CPU)",
    ("DS + MWPM", "cpu1"): "DS + MWPM (CPU)",
    ("DS + NN", "gpu"): "DS + NN (GPU)",
    ("DS + NN", "cpu1"): "DS + NN (CPU)",
}
HATCH = {("DS + NN", "cpu1"): "////"}


def shade(base_color, frac):
    """Original helper: darken with code distance."""
    rgb = np.array(mcolors.to_rgb(base_color))
    return np.clip(rgb * frac + (1 - frac), 0, 1)


def plot_runtime(df: pd.DataFrame, out_pdf: str, series=SERIES, allow_partial=False):
    d_vals = sorted(df["d"].unique())
    pT_vals = sorted(df["p_trans"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(pT_vals))
    plotted = []

    n_s, n_d = len(series), len(d_vals)
    gap = 0.7                     # space between clusters, in bar widths
    group_w = 0.88                # total width of one p_trans group
    width = group_w / (n_s * n_d + (n_s - 1) * gap)

    for j, key in enumerate(series):
        filt, dev = key
        for i_d, d in enumerate(d_vals):
            frac = 0.5 + 0.4 * (i_d / (len(d_vals) - 1))
            sub = df[(df["d"] == d) & (df["method"] == filt) & (df["device"] == dev)].sort_values("p_trans")
            if sub.empty:
                continue
            if allow_partial:                       # prototypes only (incomplete runs)
                idx = [pT_vals.index(p) for p in sub["p_trans"]]
                x_use = x[idx]
            else:
                assert list(sub["p_trans"]) == pT_vals, (key, d, list(sub["p_trans"]))
                x_use = x
            col = shade(FILTER_BASE_COLOR[filt], frac)
            slot = j * (n_d + gap) + i_d
            xpos = x_use - group_w / 2 + (slot + 0.5) * width
            ax.bar(xpos, sub["t_round_us_mean"], width=width * 0.9, color=col, alpha=0.9, zorder=2,
                   hatch=HATCH.get(key), edgecolor=("white" if key in HATCH else None), linewidth=0)
            ax.errorbar(xpos, sub["t_round_us_mean"], yerr=sub["t_round_us_std"], fmt="none",
                        ecolor="black", elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
            plotted.append(sub.assign(bar_x=xpos, bar_color=mcolors.to_hex(col), hatched=key in HATCH))

    # axes, labels (as original)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pt:.2f}" for pt in pT_vals])
    ax.set_xlabel("Transmission Error Rate", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (µs)", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # --- Legend 1: code distance (grey levels of the original legend: 0.4 + 1.5 * i / 11)
    gray_cmap = plt.cm.Greys
    gray_handles = [plt.Rectangle((0, 0), 1, 1, color=gray_cmap(0.4 + 1.5 * (i / 11)))
                    for i in range(len(d_vals))]
    leg1 = ax.legend(gray_handles, [f"d={d}" for d in d_vals], title="Code distance",
                     loc="upper left", frameon=True, bbox_to_anchor=(1.02, 1.01))
    # (original: loc="upper right", bbox_to_anchor=(1.28, 1.01); moved so that the wider filter legend,
    #  whose labels now name the device, stays outside the axes)
    ax.add_artist(leg1)

    # --- Legend 2: filter type (+ device)
    style_handles = [Patch(facecolor=FILTER_BASE_COLOR[k[0]], hatch=HATCH.get(k),
                           edgecolor=("white" if k in HATCH else FILTER_BASE_COLOR[k[0]]), linewidth=0,
                           label=SERIES_LABEL[k]) for k in series]
    leg2 = ax.legend(handles=style_handles, title="Filter + Decoder", loc="upper left", frameon=True,
                     bbox_to_anchor=(1.02, 0.52))

    plt.tight_layout()
    # check that the legends do not overlap the axes (the longer labels make the filter legend wider)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(r)
    for lg in (leg1, leg2):
        assert lg.get_window_extent(r).x0 >= ax_bb.x1, "legend overlaps the axes"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", "_preview.png"), bbox_inches="tight", dpi=110)
    cols = ["tag", "method", "device", "mode", "d", "p_trans", "t_round_us_mean", "t_round_us_std", "n_reps",
            "rounds_per_rep", "total_rounds", "total_accepted", "total_failures", "accept_frac", "tau", "valid",
            "bar_x", "bar_color", "hatched"]
    pd.concat(plotted)[cols].to_csv(out_pdf.replace(".pdf", "_plotted_values.csv"), index=False)
    plt.close(fig)
    print(f"Saved {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="A")
    ap.add_argument("--csv", default=os.path.join(HERE, "runtime_results_regen.csv"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "figs"))
    ap.add_argument("--allow-partial", action="store_true", help="prototype with incomplete data")
    args = ap.parse_args()
    plt.rcParams.update({
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.8,
    })
    df = pd.read_csv(args.csv)
    df = df[df["tag"] == args.tag]
    os.makedirs(args.outdir, exist_ok=True)
    plot_runtime(df[df["mode"] == "batch500"], os.path.join(args.outdir, "runtime.pdf"),
                 allow_partial=args.allow_partial)
    plot_runtime(df[df["mode"] == "batch1"], os.path.join(args.outdir, "runtime_latency.pdf"),
                 allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
