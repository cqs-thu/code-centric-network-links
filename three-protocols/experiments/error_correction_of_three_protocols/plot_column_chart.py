import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Patch
import json
import sys

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from cals.cal_logical_error import *

results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)
cache_file = os.path.join(results_dir, 'data_all_two_basis.csv')

try:
    df_cache = pd.read_csv(cache_file)
    print(f"--- Successfully loaded {len(df_cache)} existing data points from '{cache_file}'. ---")
    for col in ['p_local', 'p_transmit', 'logical_error']:
        if col in df_cache.columns:
            df_cache[col] = pd.to_numeric(df_cache[col], errors='coerce')
    df_cache.dropna(subset=['p_local', 'p_transmit'], inplace=True)
except FileNotFoundError:
    print(f"--- Cache file '{cache_file}' not found. A new file will be created. ---")
    df_cache = pd.DataFrame(columns=['code_type', 'p_local', 'p_transmit', 'protocol', 'logical_error'])

p_local_values = [0.0, 0.0005, 0.001, 0.002]
p_transmit_values = [0.005, 0.025, 0.045, 0.065, 0.085, 0.105, 0.125, 0.145]
code_types = ["Surface Code", "Steane Code"]
protocols = ['distillation', 'encoding', 'qecc']

protocol_map_surface = {"distillation": "distillation", "encoding": "encoding", "qecc": "QECC"}
# num_layer such that every surface code protocol has one syndrome-measurement round after transmission
surface_num_layer = {"distillation": 0, "encoding": 1, "qecc": 1}
protocol_map_steane = {"distillation": 1, "encoding": 2, "qecc": 3}

new_results = []
for code in code_types:
    for p_trans in p_transmit_values:
        for p_local in p_local_values:
            for protocol in protocols:
                exists = False
                if not df_cache.empty:
                    exists = not df_cache[
                        (df_cache['code_type'] == code) &
                        np.isclose(df_cache['p_local'], p_local) &
                        np.isclose(df_cache['p_transmit'], p_trans) &
                        (df_cache['protocol'] == protocol)
                    ].empty
                
                if exists:
                    continue

                # logical error rate = average of the Z-basis and X-basis failure probabilities
                if code == "Surface Code":
                    proto_arg = protocol_map_surface[protocol]
                    logical_error = sum(cal_surface_correction(d = 3, num_layer = surface_num_layer[protocol], p_local=p_local, p_trans=p_trans, shots = 40_000_000, protocol_type=proto_arg, basis=b) for b in "ZX") / 2
                elif code == "Steane Code":
                    proto_arg = protocol_map_steane[protocol]
                    logical_error = sum(cal_steane_correction(p_local=p_local, p_trans=p_trans, shots = 1000000, protocol_type=proto_arg, basis=b) for b in "ZX") / 2
                
                new_results.append({
                    'code_type': code, 'p_local': p_local, 'p_transmit': p_trans,
                    'protocol': protocol, 'logical_error': logical_error
                })

if new_results:
    print(f"\n--- Calculated {len(new_results)} new data points. Appending to cache... ---")
    df_new = pd.DataFrame(new_results)
    should_write_header = not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0
    df_new.to_csv(cache_file, mode='a', header=should_write_header, index=False)
    df_long = pd.concat([df_cache, df_new], ignore_index=True)
else:
    print("\n--- All data points already exist in the cache. No computation needed. ---")
    df_long = df_cache

print("\n--- Data is ready. Converting to plotting format... ---")

df = df_long.pivot_table(
    index=['code_type', 'p_local', 'p_transmit'],
    columns='protocol',
    values='logical_error'
).reset_index()


df = df.sort_values(by=["code_type", "p_local", "p_transmit"])
protocols = ['distillation', 'encoding', 'qecc']
p_local_values = sorted(df['p_local'].unique())
p_transmit_values = sorted(df['p_transmit'].unique())

protocol_palettes = {
    'qecc': sns.color_palette("Blues", n_colors=len(p_local_values)),
    'distillation': sns.color_palette("Greens", n_colors=len(p_local_values)),
    'encoding': sns.color_palette("Reds", n_colors=len(p_local_values)),
}


if __name__ == '__main__':
    plt.rcParams.update({
        "font.size": 25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.fontsize": 20,
        "axes.labelsize": 26,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "axes.titlesize": 28,
    })

    for code in df['code_type'].unique():
        df_code = df[df['code_type'] == code]
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(p_transmit_values))
        xtick_labels = [f"{pt:.3f}" for pt in p_transmit_values]

        bar_width = 0.25
        if code == "Surface Code":
            ax.set_ylim(0, 0.15017975)   # y-range of the paper figure

        for i, protocol in enumerate(protocols):
            for j, p_local in enumerate(p_local_values[::-1]):
                y_values = []
                for pt in p_transmit_values:
                    # Ensure df_code is filtered as needed before searching
                    filtered_df = df_code[
                        (df_code['p_transmit'] == pt) &
                        (df_code['p_local'] == p_local)
                    ]
                    # Check if the filtered result is empty and if the protocol column exists
                    if not filtered_df.empty and protocol in filtered_df.columns:
                        val = filtered_df[protocol].values[0]
                        y_values.append(val)
                    else:
                        y_values.append(0)

                bar_positions = x + (i - 1) * bar_width

                ax.bar(
                    bar_positions,
                    y_values,
                    width=bar_width,
                    color=protocol_palettes[protocol][j],
                    edgecolor='black',
                    zorder=2
                )


        ax.set_xlabel("Transmission Error Rate", fontweight='bold')
        ax.set_ylabel("Logical Error Rate", fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        

        protocol_legend = [
            Patch(facecolor=sns.color_palette("Greens")[len(p_local_values)//2], edgecolor='black', label='ED'),
            Patch(facecolor=sns.color_palette("Reds")[len(p_local_values)//2], edgecolor='black', label='EE-T'),
            Patch(facecolor=sns.color_palette("Blues")[len(p_local_values)//2], edgecolor='black', label='SE'),
        ]

        grays = sns.color_palette("Greys", n_colors=len(p_local_values)+2)[2:]
        gray_legend = [
            Patch(facecolor=grays[::-1][i], edgecolor='black', label=f"{p_local_values[i]}")
            for i in range(len(p_local_values))
        ]

        legend1 = ax.legend(
            handles=protocol_legend, title="Protocol", loc='upper right',
            bbox_to_anchor=(0.4, 0.98), frameon=True
        )

        legend2 = ax.legend(
            handles=gray_legend, title="Local Error", loc='upper right',
            bbox_to_anchor=(0.23, 0.98), frameon=True, ncol=1
        )

        ax.add_artist(legend1)
        ax.grid(True)
        
        save_path = os.path.join(os.path.dirname(__file__), f"{code.replace(' ', '_')}_logical_error_rate.pdf") 
        plt.savefig(save_path, dpi=300, pad_inches=0.03, bbox_inches='tight')
        print(f"Figures saved to: {save_path}")