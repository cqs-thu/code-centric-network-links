#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D


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
    key: pure | SW | DS
    """
    m = df["method"]
    if key == "pure":
        sel = m.str.fullmatch(r"(?i)pure") | m.str.contains(r"(?i)pure")
        out = df[sel].copy()
        # fallback: if file only has one curve and no explicit "pure"
        if out.empty and df["method"].nunique() == 1:
            out = df.copy()
        return out
    if key == "SW":
        return df[m.str.contains(r"\bSW\b", regex=True)].copy()
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
      data[plocal][accept_tgt]["SW"/"DS"] = curve df(p_trans, ler)
      data[plocal][1.0]["Pure"]          = curve df(p_trans, ler)  # placed at y=1.0
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
            sw = curve_mean_by_ptran(select_family(df, "SW"))
            ds = curve_mean_by_ptran(select_family(df, "DS"))
            data[pl].setdefault(a, {})
            if not sw.empty:
                data[pl][a]["SW"] = sw
            if not ds.empty:
                data[pl][a]["DS"] = ds
            continue

    return data


# -------------------------
# plotting helpers
# -------------------------
def fam_label(decoder_tag: str) -> str:
    return "MWPM" if decoder_tag.upper() == "PM" else "NN"

def safe_log10(arr: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.log10(np.maximum(arr, eps))

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
    Interpolate ler onto common x_grid (1D linear interpolation).
    Assumes df sorted by p_trans.
    """
    if df.empty or x_grid.size == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    x = df["p_trans"].values
    y = df["ler"].values
    # Only interpolate within range; outside -> NaN
    y_interp = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
    # np.interp doesn't support NaN left/right; emulate:
    y_interp[x_grid < x.min()] = np.nan
    y_interp[x_grid > x.max()] = np.nan
    return y_interp

def accept_edges(accepts_sorted: List[float]) -> List[float]:
    """
    Build band edges around discrete accept levels for terrace surfaces.
    For accepts [a0<a1<a2], edges e0<e1<e2<e3.
    Use midpoints, with symmetric extrapolation at ends.
    """
    a = accepts_sorted
    if len(a) == 1:
        # single band
        return [max(0.0, a[0] - 0.1), min(1.0, a[0] + 0.1)]
    mids = [(a[i] + a[i+1]) / 2.0 for i in range(len(a)-1)]
    e0 = a[0] - (mids[0] - a[0])
    eN = a[-1] + (a[-1] - mids[-1])
    edges = [e0] + mids + [eN]
    # clamp into [0,1]
    edges = [max(0.0, min(1.0, x)) for x in edges]
    return edges

def add_surface_legends(ax):
    """
    Keep legend minimal: show SW surface, DS surface, Pure line.
    """
    handles = [
        Line2D([0],[0], color="#1f77b4", linewidth=6, label="SW surface (terraced)"),
        Line2D([0],[0], color="#2ca02c", linewidth=6, label="DS surface (terraced)"),
        Line2D([0],[0], color="#4d4d4d", linestyle="--", marker="o", linewidth=2.2, label="Pure baseline (a=1)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)


# -------------------------
# plotting: line mode (your previous working baseline, with accept-color encoding)
# -------------------------
METHOD_STYLE = {
    "Pure": dict(linestyle="--", marker="o"),
    "SW":   dict(linestyle="-",  marker="s"),
    "DS":   dict(linestyle="-",  marker="^"),
}

def build_accept_color_map(accepts: List[float], pure_accept: float = 1.0) -> Dict[float, str]:
    cmap = plt.get_cmap("Blues")
    if len(accepts) == 1:
        levels = [0.7]
    else:
        levels = np.linspace(0.55, 0.95, num=len(accepts))
    accepts_sorted = sorted(accepts)
    acc_colors = {a: cmap(levels[i]) for i, a in enumerate(accepts_sorted)}
    acc_colors[pure_accept] = "#4d4d4d"
    return acc_colors

def plot_curve(ax, xs, ys, zs, method, acc, acc_colors, label=None):
    ax.plot(
        xs, ys, zs,
        color=acc_colors[acc],
        linewidth=2.2,
        markersize=5,
        label=label,
        **METHOD_STYLE[method],
    )


# -------------------------
# plotting: terrace surface mode (NEW)
# -------------------------
def plot_terrace_surfaces_one_panel(
    ax,
    panel_data: Dict[float, Dict[str, pd.DataFrame]],
    accepts: List[float],
    zlog: bool,
    fam: str,
    alpha_sw: float = 0.35,
    alpha_ds: float = 0.35,
    eps_y: float = 1e-4,
):
    """
    panel_data: data[plocal] mapping accept->{"SW","DS"} and 1.0->{"Pure"}
    """
    accepts_sorted = sorted(accepts)
    edges = accept_edges(accepts_sorted)  # len = len(accepts)+1

    # Collect curves for common x grid
    curves_for_x = []
    for a in accepts_sorted:
        block = panel_data.get(a, {})
        if "SW" in block: curves_for_x.append(block["SW"])
        if "DS" in block: curves_for_x.append(block["DS"])
    pure_block = panel_data.get(1.0, {}).get("Pure", pd.DataFrame())
    if not pure_block.empty:
        curves_for_x.append(pure_block)

    x_grid = build_common_x_grid(curves_for_x)
    if x_grid.size == 0:
        return

    # Helper: build terraced Y rows and Z rows for one method ("SW" or "DS")
    def build_terraced_mesh(method_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_rows = []
        z_rows = []
        for i, a in enumerate(accepts_sorted):
            block = panel_data.get(a, {})
            if method_key not in block:
                # still create band with NaN to keep structure
                z_a = np.full_like(x_grid, np.nan, dtype=float)
            else:
                z_a = interp_curve_to_x(block[method_key], x_grid)
                z_a = safe_log10(z_a) if zlog else z_a

            # band in y: [edge_i, edge_{i+1}] but use +-eps to create vertical step walls
            y_low = edges[i]
            y_high = edges[i+1]
            # four rows: (low), (high-eps) share same z; then (high+eps) next band will use its z
            y_rows.append(y_low)
            z_rows.append(z_a)
            y_rows.append(min(1.0, y_high - eps_y))
            z_rows.append(z_a)

            # insert an immediate jump row at y_high+eps with the *next* band's z to create the vertical face
            # (skip for last band)
            if i < len(accepts_sorted) - 1:
                y_rows.append(min(1.0, y_high + eps_y))
                # next band's z will be added when loop advances; for the wall row we can repeat current,
                # because the next band starts at y_high+eps with its own z. Here we just create the coordinate.
                z_rows.append(z_a)

        Y = np.array(y_rows, dtype=float)
        Z = np.vstack(z_rows)  # (ny, nx)
        X = np.tile(x_grid[None, :], (Z.shape[0], 1))
        Y2 = np.tile(Y[:, None], (1, x_grid.size))
        return X, Y2, Z

    # Build and plot SW surface
    Xs, Ys, Zs = build_terraced_mesh("SW")
    ax.plot_surface(
        Xs, Ys, Zs,
        color="#1f77b4",  # blue
        alpha=alpha_sw,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )

    # Build and plot DS surface
    Xd, Yd, Zd = build_terraced_mesh("DS")
    ax.plot_surface(
        Xd, Yd, Zd,
        color="#2ca02c",  # green
        alpha=alpha_ds,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )

    # Optional: pure baseline line at y=1.0
    if not pure_block.empty:
        xs = pure_block["p_trans"].values
        ys = np.full_like(xs, 1.0, dtype=float)
        zs = safe_log10(pure_block["ler"].values) if zlog else pure_block["ler"].values
        ax.plot(xs, ys, zs, color="#4d4d4d", linestyle="--", marker="o", linewidth=2.2, markersize=4, label=f"Pure {fam}")

    # Axis labels
    ax.set_xlabel(r"$p_{\mathrm{trans}}$")
    ax.set_ylabel(r"$a_{\mathrm{tgt}}$")
    ax.set_zlabel(r"$\log_{10}(p_L)$" if zlog else r"$p_L$")

    # y ticks show accept levels and 1.0
    ax.set_yticks(sorted(list(set([1.0] + accepts_sorted))))

    # Light grid
    ax.grid(True)


# -------------------------
# master plot function: mode switch
# -------------------------
def plot_3d_for_decoder(
    data_dir: Path,
    decoder_tag: str,
    d: int,
    plocals: List[float],
    accepts: List[float],
    out_path: Path,
    mode: str = "line",   # "line" or "surface"
    zlog: bool = True,
    elev: float = 25,
    azim: float = -55,
    surface_alpha: float = 0.35,
    z_offset_ds: float = 0.0
):
    data = load_all(data_dir, decoder_tag, d)

    plocals_present = [pl for pl in plocals if pl in data]
    if not plocals_present:
        raise ValueError(f"Requested plocals not found for {decoder_tag}: {plocals}")

    n = len(plocals_present)
    fig = plt.figure(figsize=(6.2 * n, 5.8))

    fam = fam_label(decoder_tag)

    # For line mode accept-color map
    acc_colors = build_accept_color_map(accepts, pure_accept=1.0)

    for idx, pl in enumerate(plocals_present, start=1):
        ax = fig.add_subplot(1, n, idx, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        panel_data = data[pl]

        if mode == "surface":
            plot_terrace_surfaces_one_panel(
                ax=ax,
                panel_data=panel_data,
                accepts=accepts,
                zlog=zlog,
                fam=fam,
                alpha_sw=surface_alpha,
                alpha_ds=surface_alpha,
            )
            ax.set_title(rf"{fam} terraced surfaces, $p_{{\mathrm{{local}}}}={pl:g}$")
            if idx == 1:
                add_surface_legends(ax)
        elif mode == "surface_lines":
            plot_surface_and_lines_one_panel(
                ax=ax,
                panel_data=panel_data,
                accepts=accepts,
                zlog=zlog,
                fam=fam,
                surface_alpha=surface_alpha,
                wire_alpha=0.35,
                z_offset_ds=0,
            )
            ax.set_title(rf"{fam} surface+lines, $p_{{\mathrm{{local}}}}={pl:g}$")

        elif mode == "line":
            # ---- Pure (a=1.0) ----
            pure_block = panel_data.get(1.0, {}).get("Pure", pd.DataFrame())
            if not pure_block.empty:
                xs = pure_block["p_trans"].values
                ys = np.full_like(xs, 1.0, dtype=float)
                zs = safe_log10(pure_block["ler"].values) if zlog else pure_block["ler"].values
                plot_curve(ax, xs, ys, zs, method="Pure", acc=1.0, acc_colors=acc_colors, label=f"Pure {fam}")

            # ---- SW/DS at each accept target (same accept -> same color) ----
            for a in sorted(accepts):
                block = panel_data.get(a, {})
                for key in ["SW", "DS"]:
                    if key not in block or block[key].empty:
                        continue
                    cdf = block[key]
                    xs = cdf["p_trans"].values
                    ys = np.full_like(xs, a, dtype=float)
                    zs = safe_log10(cdf["ler"].values) if zlog else cdf["ler"].values
                    plot_curve(ax, xs, ys, zs, method=key, acc=a, acc_colors=acc_colors, label=None)

            ax.set_xlabel(r"$p_{\mathrm{trans}}$")
            ax.set_ylabel(r"$a_{\mathrm{tgt}}$")
            ax.set_zlabel(r"$\log_{10}(p_L)$" if zlog else r"$p_L$")
            ax.set_title(rf"{fam} lines, $p_{{\mathrm{{local}}}}={pl:g}$")
            ax.set_yticks(sorted(list(set([1.0] + accepts))))
            ax.grid(True)

            # Minimal legend only on first panel
            if idx == 1:
                handles = [
                    Line2D([0],[0], color="#4d4d4d", linestyle="--", marker="o", linewidth=2.2, label=f"Pure {fam}"),
                    Line2D([0],[0], color=acc_colors[min(accepts)], linestyle="-", marker="s", linewidth=2.2, label="SW (color by a_tgt)"),
                    Line2D([0],[0], color=acc_colors[min(accepts)], linestyle="-", marker="^", linewidth=2.2, label="DS (color by a_tgt)"),
                ]
                ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_lines_with_accept_depth_one_panel(
    ax,
    panel_data: Dict[float, Dict[str, pd.DataFrame]],
    accepts: List[float],
    zlog: bool,
    fam: str,
    band: bool = True,
    band_width: float = 0.06,   # y-direction band width around each accept line
    band_alpha: float = 0.10,   # very light
    elev: float = 25,
):
    """
    Draw all SW/DS curves as lines.
    - SW: blue colormap by accept
    - DS: green colormap by accept
    - optional: narrow band surface around each accept to keep a "terrace" feeling
    """

    accepts_sorted = sorted(accepts)

    # Colormaps for accept depth
    blues = plt.get_cmap("Blues")
    greens = plt.get_cmap("Greens")
    # sample from mid-range to avoid too pale / too dark
    if len(accepts_sorted) == 1:
        levels = [0.7]
    else:
        levels = np.linspace(0.55, 0.85, num=len(accepts_sorted))
    sw_color = {a: blues(levels[i]) for i, a in enumerate(accepts_sorted)}
    ds_color = {a: greens(levels[i]) for i, a in enumerate(accepts_sorted)}

    # ---- Pure baseline (a=1) ----
    pure_block = panel_data.get(1.0, {}).get("Pure", pd.DataFrame())
    if not pure_block.empty:
        xs = pure_block["p_trans"].values
        ys = np.full_like(xs, 1.0, dtype=float)
        zs = safe_log10(pure_block["ler"].values) if zlog else pure_block["ler"].values
        ax.plot(xs, ys, zs, color="#4d4d4d", linestyle="--", marker="o",
                linewidth=2.2, markersize=4, label=f"Pure {fam} (a=1)")

    # helper for optional band (narrow surface strip)
    def plot_band(xs: np.ndarray, a: float, zs: np.ndarray, color, alpha: float):
        # build a 2-row strip in y
        y0 = max(0.0, a - band_width / 2.0)
        y1 = min(1.0, a + band_width / 2.0)
        X = np.vstack([xs, xs])
        Y = np.vstack([np.full_like(xs, y0), np.full_like(xs, y1)])
        Z = np.vstack([zs, zs])
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0.0, antialiased=True, shade=True)

    # ---- SW/DS lines per accept ----
    for a in accepts_sorted:
        block = panel_data.get(a, {})

        # SW
        if "SW" in block and not block["SW"].empty:
            cdf = block["SW"]
            xs = cdf["p_trans"].values
            ys = np.full_like(xs, a, dtype=float)
            zs = safe_log10(cdf["ler"].values) if zlog else cdf["ler"].values
            ax.plot(xs, ys, zs, color=sw_color[a], linestyle="-", marker="s",
                    linewidth=2.2, markersize=4)
            if band:
                plot_band(xs, a, zs, color=sw_color[a], alpha=band_alpha)

        # DS
        if "DS" in block and not block["DS"].empty:
            cdf = block["DS"]
            xs = cdf["p_trans"].values
            ys = np.full_like(xs, a, dtype=float)
            zs = safe_log10(cdf["ler"].values) if zlog else cdf["ler"].values
            ax.plot(xs, ys, zs, color=ds_color[a], linestyle="-", marker="^",
                    linewidth=2.2, markersize=4)
            if band:
                plot_band(xs, a, zs, color=ds_color[a], alpha=band_alpha)

    # Axes
    ax.set_xlabel(r"$p_{\mathrm{trans}}$")
    ax.set_ylabel(r"$a_{\mathrm{tgt}}$")
    ax.set_zlabel(r"$\log_{10}(p_L)$" if zlog else r"$p_L$")
    ax.set_yticks(sorted(list(set([1.0] + accepts_sorted))))
    ax.grid(True)

    # Two legends: method + accept depth (as colorbars-like legend)
    method_handles = [
        Line2D([0],[0], color="#4d4d4d", linestyle="--", marker="o", linewidth=2.2, label="Pure"),
        Line2D([0],[0], color=blues(0.7), linestyle="-", marker="s", linewidth=2.2, label="SW (blue)"),
        Line2D([0],[0], color=greens(0.7), linestyle="-", marker="^", linewidth=2.2, label="DS (green)"),
    ]
    leg1 = ax.legend(handles=method_handles, title="Method", loc="upper left",
                     fontsize=9, frameon=True, title_fontsize=9)
    ax.add_artist(leg1)

    # Accept legend: show three swatches for accept values (use SW blue shades for simplicity)
    acc_handles = []
    for i, a in enumerate(accepts_sorted):
        acc_handles.append(Line2D([0],[0], color=sw_color[a], linewidth=4, label=rf"$a_{{\mathrm{{tgt}}}}={a:g}$"))
    ax.legend(handles=acc_handles, title="Target accept (shade)", loc="upper right",
              fontsize=9, frameon=True, title_fontsize=9)

def build_mesh_for_method(panel_data, accepts, method_key, x_grid, zlog):
    """
    Build Z matrix of shape (len(accepts), len(x_grid)) for method SW/DS.
    Only uses discrete accept levels (no interpolation in y).
    """
    Z_rows = []
    for a in sorted(accepts):
        block = panel_data.get(a, {})
        if method_key not in block or block[method_key].empty:
            z = np.full_like(x_grid, np.nan, dtype=float)
        else:
            z = interp_curve_to_x(block[method_key], x_grid)
            z = safe_log10(z) if zlog else z
        Z_rows.append(z)
    Z = np.vstack(Z_rows)
    return Z  # (ny, nx)

def plot_surface_and_lines_one_panel(
    ax,
    panel_data: Dict[float, Dict[str, pd.DataFrame]],
    accepts: List[float],
    zlog: bool,
    fam: str,
    surface_alpha: float = 0.22,
    wire_alpha: float = 0.35,
    z_offset_ds: float = 0.0,   # tiny offset to separate two surfaces visually
    break_even_plane: bool = True,
    break_even_alpha: float = 0.5,
):
    """
    Draw:
      - SW surface (blue, with wireframe edges)
      - DS surface (green, without edges)
      - All SW/DS lines (accept shaded)
      - Pure baseline at a=1 (grey dashed)
      - Break-even plane: p_L = 2 p_trans / 3  (very light gray)
    """

    accepts_sorted = sorted(accepts)

    # Build common x-grid across SW/DS (and optionally pure)
    curves_for_x = []
    for a in accepts_sorted:
        block = panel_data.get(a, {})
        if "SW" in block: curves_for_x.append(block["SW"])
        if "DS" in block: curves_for_x.append(block["DS"])
    pure_block = panel_data.get(1.0, {}).get("Pure", pd.DataFrame())
    if not pure_block.empty:
        curves_for_x.append(pure_block)

    x_grid = build_common_x_grid(curves_for_x)
    if x_grid.size == 0:
        return

    # Mesh coordinates (discrete y grid for surfaces)
    Y_levels = np.array(accepts_sorted, dtype=float)
    X, Y = np.meshgrid(x_grid, Y_levels)

    # ---- Break-even plane: p_L = 2 p_trans / 3 (an unencoded |0> read out in Z) ----
    # In zlog mode, z = log10(2 p_trans / 3); otherwise z = 2 p_trans / 3.
    if break_even_plane:
        # Use a y-grid that also spans up to 1.0, so the plane covers the pure baseline line as well.
        y_plane_levels = np.array(sorted(list(set(accepts_sorted + [1.0]))), dtype=float)
        Xp, Yp = np.meshgrid(x_grid, y_plane_levels)
        Zp = safe_log10(2 * Xp / 3) if zlog else 2 * Xp / 3

        ax.plot_surface(
            Xp, Yp, Zp,
            color="#bdbdbd",        # very light gray
            alpha=break_even_alpha, # very transparent
            linewidth=0.0,
            edgecolor="none",
            antialiased=True,
            shade=False,            # keep it neutral (no lighting artifacts)
            zorder=0,
        )

    # ---- Z for SW/DS ----
    Z_sw = build_mesh_for_method(panel_data, accepts_sorted, "SW", x_grid, zlog)
    Z_ds = build_mesh_for_method(panel_data, accepts_sorted, "DS", x_grid, zlog)

    # ---- Draw surfaces ----
    # SW: blue with subtle edges (wire texture)
    ax.plot_surface(
        X, Y, Z_sw,
        color="#1f77b4",
        alpha=surface_alpha,
        linewidth=0.6,
        edgecolor=None,   # subtle dark edges
        antialiased=True,
        shade=True,
    )

    # DS: green, no edges
    ax.plot_surface(
        X, Y, Z_ds + z_offset_ds,
        color="#2ca02c",
        alpha=surface_alpha,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        shade=True,
    )

    # ---- Lines (accept shaded) ----
    blues = plt.get_cmap("Blues")
    greens = plt.get_cmap("Greens")
    if len(accepts_sorted) == 1:
        levels = [0.7]
    else:
        levels = np.linspace(0.65, 0.95, num=len(accepts_sorted))
    sw_color = {a: blues(levels[i]) for i, a in enumerate(accepts_sorted)}
    ds_color = {a: greens(levels[i]) for i, a in enumerate(accepts_sorted)}

    # Pure baseline (a=1)
    if not pure_block.empty:
        xs = pure_block["p_trans"].values
        ys = np.full_like(xs, 1.0, dtype=float)
        zs = safe_log10(pure_block["ler"].values) if zlog else pure_block["ler"].values
        ax.plot(xs, ys, zs, color="#4d4d4d", linestyle="--", marker="o", linewidth=2.2, markersize=4)

    for a in accepts_sorted:
        block = panel_data.get(a, {})

        # SW line
        if "SW" in block and not block["SW"].empty:
            cdf = block["SW"]
            xs = cdf["p_trans"].values
            ys = np.full_like(xs, a, dtype=float)
            zs = safe_log10(cdf["ler"].values) if zlog else cdf["ler"].values
            ax.plot(xs, ys, zs, color=sw_color[a], linestyle="-", marker="s", linewidth=2.2, markersize=4)

        # DS line
        if "DS" in block and not block["DS"].empty:
            cdf = block["DS"]
            xs = cdf["p_trans"].values
            ys = np.full_like(xs, a, dtype=float)
            zs = safe_log10(cdf["ler"].values) if zlog else cdf["ler"].values
            ax.plot(xs, ys, zs, color=ds_color[a], linestyle="-", marker="^", linewidth=2.2, markersize=4)

    # ---- Labels / ticks ----
    ax.set_xlabel(r"$p_{\mathrm{trans}}$")
    ax.set_ylabel(r"$a_{\mathrm{tgt}}$")
    ax.set_zlabel(r"$\log_{10}(p_L)$" if zlog else r"$p_L$")
    ax.set_yticks(sorted(list(set([1.0] + accepts_sorted))))
    ax.grid(True)

    # ---- Legends: method + accept shade ----
    method_handles = [
        Line2D([0],[0], color="#4d4d4d", linestyle="--", marker="o", linewidth=2.2, label="Pure"),
        Line2D([0],[0], color=blues(0.7), linestyle="-", marker="s", linewidth=2.2, label="SW (blue surface/lines)"),
        Line2D([0],[0], color=greens(0.7), linestyle="-", marker="^", linewidth=2.2, label="DS (green surface/lines)"),
    ]
    if break_even_plane:
        method_handles.append(Line2D([0],[0], color="#bdbdbd", linewidth=4, label="Break-even plane ($p_L=2p_{trans}/3$)"))

    leg1 = ax.legend(handles=method_handles, title="Objects", loc="upper left",
                     fontsize=9, frameon=True, title_fontsize=9)
    ax.add_artist(leg1)

    acc_handles = [Line2D([0],[0], color=sw_color[a], linewidth=4, label=rf"$a_{{\mathrm{{tgt}}}}={a:g}$")
                   for a in accepts_sorted]
    ax.legend(handles=acc_handles, title="Target accept (shade)", loc="upper right",
              fontsize=9, frameon=True, title_fontsize=9)


def main():
    ap = argparse.ArgumentParser(description="3D plots for (p_trans, a_tgt, LER) with facets over p_local.")
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--out_dir", default="fig_out", type=str)
    ap.add_argument("--d", default=3, type=int)
    ap.add_argument("--plocals", default="0.01,0.005,0.001", type=str)
    ap.add_argument("--accepts", default="0.2,0.5,0.8", type=str)
    ap.add_argument("--mode", default="surface_lines",
                choices=["line", "surface", "surface_lines"],
                help="Plot mode.")
    ap.add_argument("--z_offset_ds", default=0.0, type=float,
                help="Tiny z offset added to DS surface (e.g., 0.01) to improve separation.")
    ap.add_argument("--no_zlog", action="store_true", help="Use linear z instead of log10.")
    ap.add_argument("--elev", default=25.0, type=float)
    ap.add_argument("--azim", default=-55.0, type=float)
    ap.add_argument("--surface_alpha", default=0.35, type=float, help="Alpha for SW/DS surfaces in surface mode.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    d = args.d
    plocals = [float(x.strip()) for x in args.plocals.split(",") if x.strip()]
    accepts = [float(x.strip()) for x in args.accepts.split(",") if x.strip()]

    plocals = sorted(plocals, reverse=True)
    accepts = sorted(accepts)

    zlog = (not args.no_zlog)

    plot_3d_for_decoder(
        data_dir=data_dir,
        decoder_tag="PM",
        d=d,
        plocals=plocals,
        accepts=accepts,
        out_path=out_dir / f"Fig3D_PM_{args.mode}_d{d}.png",
        mode=args.mode,
        zlog=zlog,
        elev=args.elev,
        azim=args.azim,
        surface_alpha=args.surface_alpha,
        z_offset_ds=args.z_offset_ds,
    )
    plot_3d_for_decoder(
        data_dir=data_dir,
        decoder_tag="NN",
        d=d,
        plocals=plocals,
        accepts=accepts,
        out_path=out_dir / f"Fig3D_NN_{args.mode}_d{d}.png",
        mode=args.mode,
        zlog=zlog,
        elev=args.elev,
        azim=args.azim,
        surface_alpha=args.surface_alpha,
        z_offset_ds=args.z_offset_ds,
    )

    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
