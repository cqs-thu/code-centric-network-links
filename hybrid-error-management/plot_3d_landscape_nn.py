#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_3d_landscape_nn.py
3D landscape of the DS + NN logical error rate over (transmission error rate,
target acceptance rate), one panel per local error rate (Figure 8).

Reads the NN caches written by near_term_smallcode.py:
  NN_d{d}_plocal{p}_pure.csv        unfiltered NN decoder, drawn at acceptance 1.0
  NN_d{d}_plocal{p}_accept{a}.csv   DS + NN at target acceptance a (the SW rows are not used)
Default data: cache_nearterm (run near_term_smallcode.py first).

Usage:
  python near_term_smallcode.py
  python plot_3d_landscape_nn.py
  python plot_3d_landscape_nn.py --data_dir cache_nearterm --out_dir fig_out
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "cache_nearterm")
OUT_DIR = os.path.join(HERE, "fig_out")


# -------------------------
# filename parsing
# -------------------------
ACCEPT_RE = re.compile(r"_accept([0-9]*\.?[0-9]+)\.csv$")
PURE_RE   = re.compile(r"_pure\.csv$")
PLOCAL_RE = re.compile(r"_plocal([0-9]*\.?[0-9]+)")

def parse_plocal_from_name(name: str) -> float:
    m = PLOCAL_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse plocal from filename: {name}")
    return float(m.group(1))

def parse_accept_from_name(name: str) -> float:
    m = ACCEPT_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse accept from filename: {name}")
    return float(m.group(1))

def is_pure_file(name: str) -> bool:
    return PURE_RE.search(name) is not None

def is_accept_file(name: str) -> bool:
    return ACCEPT_RE.search(name) is not None


# -------------------------
# csv parsing
# -------------------------
def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"method", "p_trans", "ler"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    df = df.copy()
    df["p_trans"] = pd.to_numeric(df["p_trans"], errors="coerce")
    df["ler"] = pd.to_numeric(df["ler"], errors="coerce")
    df = df.dropna(subset=["p_trans", "ler"])
    df["method"] = df["method"].astype(str)
    return df

def select_family(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    key: pure | DS
    """
    m = df["method"]
    if key == "pure":
        out = df[m.str.contains(r"(?i)pure")].copy()
        # a pure file holds a single curve (method "unfiltered")
        if out.empty and df["method"].nunique() == 1:
            out = df.copy()
        return out
    if key == "DS":
        return df[m.str.contains(r"\bDS\b", regex=True)].copy()
    raise ValueError(key)

def curve_mean_by_ptran(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby("p_trans", as_index=False)["ler"].mean()
    return g.sort_values("p_trans").reset_index(drop=True)


# -------------------------
# data assembly
# -------------------------
def load_all(data_dir: Path, decoder_tag: str, d: int) -> Dict[float, Dict[float, Dict[str, pd.DataFrame]]]:
    """
    returns:
      data[plocal][accept]["DS"] = curve df(p_trans, ler)
      data[plocal][1.0]["Pure"]  = curve df(p_trans, ler)  # placed at y=1.0
    """
    pattern = f"{decoder_tag}_d{d}_plocal"
    files = sorted([p for p in data_dir.glob(f"{pattern}*.csv")])

    if not files:
        raise FileNotFoundError(f"No files match {pattern}*.csv in {data_dir}")

    data: Dict[float, Dict[float, Dict[str, pd.DataFrame]]] = {}

    for path in files:
        name = path.name
        pl = parse_plocal_from_name(name)
        data.setdefault(pl, {})

        df = read_csv(path)

        if is_pure_file(name):
            pure = curve_mean_by_ptran(select_family(df, "pure"))
            data[pl].setdefault(1.0, {})["Pure"] = pure
            continue

        if is_accept_file(name):
            a = parse_accept_from_name(name)
            ds = curve_mean_by_ptran(select_family(df, "DS"))
            data[pl].setdefault(a, {})
            if not ds.empty:
                data[pl][a]["DS"] = ds
            continue

    return data


# -------------------------
# plotting helpers
# -------------------------
def build_common_x_grid(curves: List[pd.DataFrame]) -> np.ndarray:
    xs = []
    for df in curves:
        if df is None or df.empty:
            continue
        xs.append(df["p_trans"].values)
    if not xs:
        return np.array([])
    x = np.unique(np.concatenate(xs))
    return np.sort(x)

def interp_curve_to_x(df: pd.DataFrame, x_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate ler onto the common x_grid (1D linear interpolation); NaN outside the data range.
    Assumes df sorted by p_trans.
    """
    if df.empty or x_grid.size == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    x = df["p_trans"].values
    y = df["ler"].values
    y_interp = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
    y_interp[x_grid < x.min()] = np.nan
    y_interp[x_grid > x.max()] = np.nan
    return y_interp


# -------------------------
# one panel: surface + lines
# -------------------------
def plot_surface_and_lines_one_panel(
    ax,
    panel_data: Dict[float, Dict[str, pd.DataFrame]],
    accepts: List[float],
    surface_alpha: float = 0.35,
    break_even_alpha: float = 0.65,
    show_legend: bool = False,
):
    """
    Draw:
      - Break-even plane p_L = 2 p_trans / 3 (gray): an unencoded |0> read out in Z fails with this probability
      - DS surface (green) over the target accepts, with the top row (acceptance 1.0) from the Pure curve
      - Lines for all acceptance levels (darkest green for 1.0)
    Returns the legend handles if show_legend is True.
    """
    accepts_sorted = sorted(accepts)

    # Collect curves to build the common x-grid
    curves_for_x = []
    for a in accepts_sorted:
        block = panel_data.get(a, {})
        if "DS" in block: curves_for_x.append(block["DS"])
    pure_block = panel_data.get(1.0, {}).get("Pure", pd.DataFrame())
    if not pure_block.empty:
        curves_for_x.append(pure_block)

    x_grid = build_common_x_grid(curves_for_x)
    if x_grid.size == 0:
        return None

    # y levels include the accepts plus 1.0, so the DS surface connects to Pure at the top
    y_levels = np.array(sorted(list(set(accepts_sorted + [1.0]))))
    X, Y = np.meshgrid(x_grid, y_levels)

    Z_rows = []
    for y in y_levels:
        if np.isclose(y, 1.0) and not pure_block.empty:
            z = interp_curve_to_x(pure_block, x_grid)
        else:
            block = panel_data.get(float(y), {})
            if "DS" not in block or block["DS"].empty:
                z = np.full_like(x_grid, np.nan, dtype=float)
            else:
                z = interp_curve_to_x(block["DS"], x_grid)
        Z_rows.append(z)
    Z = np.vstack(Z_rows)

    # Break-even plane p_L = 2 p_trans / 3
    Xp, Yp = np.meshgrid(x_grid, y_levels)
    ax.plot_surface(
        Xp, Yp, 2 * Xp / 3,
        color="#bdbdbd",
        alpha=break_even_alpha,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        shade=False,
        zorder=0,
    )

    # Green shades by acceptance level (darkest for 1.0)
    greens = plt.get_cmap("Greens")
    all_levels = sorted(list(set([1.0] + accepts_sorted)))
    shade_vals = np.linspace(0.45, 0.9, num=max(len(all_levels), 2))
    color_map = {lvl: greens(shade_vals[i]) for i, lvl in enumerate(all_levels)}
    color_map[1.0] = greens(0.95)

    # DS surface
    ax.plot_surface(
        X, Y, Z,
        color="#2ca02c",
        alpha=surface_alpha,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        shade=True,
    )

    # Lines for all acceptance levels (including 1.0)
    for a in all_levels:
        if np.isclose(a, 1.0) and not pure_block.empty:
            xs = pure_block["p_trans"].values
            ys = np.full_like(xs, 1.0, dtype=float)
            zs = pure_block["ler"].values
            ax.plot(xs, ys, zs, color=color_map[1.0], linestyle="--", marker="o",
                    linewidth=2.2, markersize=4)
        else:
            block = panel_data.get(a, {})
            if "DS" in block and not block["DS"].empty:
                cdf = block["DS"]
                xs = cdf["p_trans"].values
                ys = np.full_like(xs, a, dtype=float)
                zs = cdf["ler"].values
                ax.plot(xs, ys, zs, color=color_map[a], linestyle="-", marker="^",
                        linewidth=2.2, markersize=4)

    # Axes/ticks
    ax.set_xlabel("Transmission Error Rate", fontweight="bold", fontsize=13)
    ax.set_ylabel("Target Accept Rate", fontweight="bold", fontsize=13)
    ax.set_zlabel("Logical Error Rate", fontweight="bold", fontsize=13)
    ax.set_yticks(sorted(all_levels))
    ax.set_zlim(0.0, 0.33)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True)

    # Legend handles, drawn once at figure level in two columns: [1.0, 0.8, break-even], [0.5, 0.2]
    if show_legend:
        return [
            Line2D([0],[0], color=color_map.get(1.0, greens(0.95)), linewidth=4, label=r"$a_{\mathrm{accept}}=1.0$"),
            Line2D([0],[0], color=color_map.get(0.8, greens(0.85)), linewidth=4, label=r"$a_{\mathrm{accept}}=0.8$"),
            Line2D([0],[0], color="#bdbdbd", linewidth=4, label=r"Break-even plane"),
            Line2D([0],[0], color=color_map.get(0.5, greens(0.75)), linewidth=4, label=r"$a_{\mathrm{accept}}=0.5$"),
            Line2D([0],[0], color=color_map.get(0.2, greens(0.65)), linewidth=4, label=r"$a_{\mathrm{accept}}=0.2$"),
        ]
    return None


# -------------------------
# figure: one panel per p_local
# -------------------------
def plot_3d_nn(
    data_dir: Path,
    d: int,
    plocals: List[float],
    accepts: List[float],
    out_path: Path,
    elev: float = 25,
    azim: float = -70,
    surface_alpha: float = 0.35,
):
    data = load_all(data_dir, "NN", d)

    plocals_present = [pl for pl in plocals if pl in data]
    if not plocals_present:
        raise ValueError(f"Requested plocals not found for NN: {plocals}")

    n = len(plocals_present)
    fig = plt.figure(figsize=(6.2 * n, 5.8))
    axes = []
    legend_handles = None

    for idx, pl in enumerate(plocals_present, start=1):
        ax = fig.add_subplot(1, n, idx, projection="3d")
        axes.append(ax)
        ax.view_init(elev=elev, azim=azim)

        handles = plot_surface_and_lines_one_panel(
            ax=ax,
            panel_data=data[pl],
            accepts=accepts,
            surface_alpha=surface_alpha,
            show_legend=(idx == 1),
        )
        if legend_handles is None and handles is not None:
            legend_handles = handles
        # panel title as axes text (a 3D title would be moved by tight_layout)
        ax.set_title("")
        ax.text2D(
            0.5, 0.92, f"Local Error Rate = {pl:g}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontweight="bold",
            fontsize=13,
        )

    fig.tight_layout(rect=[-0.02, -0.05, 0.99, 0.99])
    fig.subplots_adjust(right=0.98, top=1)

    # Legend between the first and second panel (figure coordinates)
    if legend_handles is not None and len(axes) >= 2:
        pos1 = axes[0].get_position()
        pos2 = axes[1].get_position()
        x_mid_gap = 0.5 * (pos1.x1 + pos2.x0) + 0.001
        y_top = max(pos1.y1, pos2.y1) - 0.06

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(x_mid_gap, y_top),
            bbox_transform=fig.transFigure,
            ncol=2,
            fontsize=11,
            frameon=True,
            columnspacing=1.2,
            handlelength=2.5,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="3D DS + NN landscape (p_trans, a_accept, LER) with one panel per p_local.")
    ap.add_argument("--data_dir", default=DATA_DIR, type=str, help="directory with NN_d{d}_plocal*.csv (default: cache_nearterm)")
    ap.add_argument("--out_dir", default=OUT_DIR, type=str, help="output directory (default: fig_out)")
    ap.add_argument("--d", default=3, type=int)
    ap.add_argument("--plocals", default="0.01,0.005,0.001", type=str)
    ap.add_argument("--accepts", default="0.2,0.5,0.8", type=str)
    ap.add_argument("--elev", default=25.0, type=float)
    ap.add_argument("--azim", default=-70.0, type=float)
    ap.add_argument("--surface_alpha", default=0.35, type=float, help="Alpha of the DS surface.")
    args = ap.parse_args()

    plocals = sorted([float(x.strip()) for x in args.plocals.split(",") if x.strip()], reverse=True)
    accepts = sorted([float(x.strip()) for x in args.accepts.split(",") if x.strip()])

    out_path = Path(args.out_dir) / f"Fig3D_NN_surface_lines_d{args.d}.pdf"
    plot_3d_nn(
        data_dir=Path(args.data_dir),
        d=args.d,
        plocals=plocals,
        accepts=accepts,
        out_path=out_path,
        elev=args.elev,
        azim=args.azim,
        surface_alpha=args.surface_alpha,
    )
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    main()
