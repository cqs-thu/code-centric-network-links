import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from cals.cal_logical_error import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Patch

def get_cached_result(p_local, p_trans, cache_file_path):
    if not os.path.exists(cache_file_path) or os.path.getsize(cache_file_path) == 0:
        return None
    
    cache_df = pd.read_csv(cache_file_path)
    res = cache_df[(cache_df['p_local'] == p_local) & (cache_df['p_transmission'] == p_trans)]
    
    if not res.empty:
        encoding_col_name = 'encoding'
        protocol_2g_col_name = 'protocol_2g'
        
        if encoding_col_name in res.columns and protocol_2g_col_name in res.columns:
            return res.iloc[0][encoding_col_name], res.iloc[0][protocol_2g_col_name]
    return None

def add_result_to_cache(p_local, p_trans, encoding_logical_error, protocol_2g_logical_error, cache_file_path):
    columns = ['p_local', 'p_transmission', 'encoding', 'protocol_2g']
    new_data = pd.DataFrame([{
        'p_local': p_local, 'p_transmission': p_trans,
        'encoding': encoding_logical_error, 'protocol_2g': protocol_2g_logical_error
    }])
    
    if os.path.exists(cache_file_path) and os.path.getsize(cache_file_path) > 0:
        new_data.to_csv(cache_file_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(cache_file_path, mode='w', header=True, index=False)


def prepare_data(p_local_list, p_transmission_list, code_type = "surface"):
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    cache_file = os.path.join(results_dir, f'{code_type}_comparison_two_basis_cache.csv')

    all_results = []
    for p_local in p_local_list:
        for p_trans in p_transmission_list:
            
            cached_res = get_cached_result(p_local, p_trans, cache_file)
            
            if cached_res:
                print(f"Using cache for: pl={p_local}, pt={p_trans}")
                encoding_error, protocol_2g_error = cached_res
            else:
                print(f"Calculating for: pl={p_local}, pt={p_trans}")
                # logical error rate = average of the Z-basis and X-basis failure probabilities
                if code_type == "surface":
                    encoding_error = sum(cal_surface_correction(d=3, num_layer=1, p_local=p_local, p_trans=p_trans, shots = 40_000_000, protocol_type="encoding", basis=b) for b in "ZX") / 2
                    protocol_2g_error = sum(cal_2g_surface_correction(d=3, p_local=p_local, p_trans=p_trans, n=40_000_000, basis=b) for b in "ZX") / 2
                elif code_type == "steane":
                    encoding_error = sum(cal_steane_correction(p_local=p_local, p_trans=p_trans, shots = 1000000, protocol_type=2, basis=b) for b in "ZX") / 2
                    protocol_2g_error = sum(cal_2g_steane_correction(p_local=p_local, p_trans=p_trans, shots = 1000000, basis=b) for b in "ZX") / 2
                add_result_to_cache(p_local, p_trans, encoding_error, protocol_2g_error, cache_file)
            
            all_results.append({'p_local': p_local, 'p_transmission': p_trans, 'protocol': 'Encoding', 'logical_error': encoding_error})
            all_results.append({'p_local': p_local, 'p_transmission': p_trans, 'protocol': 'Protocol_2G', 'logical_error': protocol_2g_error})
            
    return pd.DataFrame(all_results)


def create_comparison_bar_chart(df, y_col, y_label, title, output_filename):
    p_local_values = sorted(df['p_local'].unique())
    p_transmit_values = sorted(df['p_transmission'].unique())
    protocols = ['Encoding', 'Protocol_2G']

    protocol_palettes = {
        'Encoding': sns.color_palette("Greens_r", n_colors=len(p_local_values)),
        'Protocol_2G': sns.color_palette("Reds_r", n_colors=len(p_local_values)),
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(p_transmit_values))
    xtick_labels = [f"{pt:.3f}" for pt in p_transmit_values]
    bar_width = 0.35 
    
    loop_indices = range(len(p_local_values) - 1, -1, -1)

    for i, protocol in enumerate(protocols):
        bar_positions = x + (i - 0.5) * bar_width
        
        for j in loop_indices:
            p_local = p_local_values[j]
            y_values = df[
                (df['protocol'] == protocol) & (df['p_local'] == p_local)
            ].sort_values('p_transmission')[y_col].values
            color = protocol_palettes[protocol][j]
            ax.bar(bar_positions, y_values, width=bar_width, color=color, edgecolor='black', zorder=2)

    ax.set_xlabel("Transmission Error Rate", fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    # ax.set_yscale('log')
    if 'surface' in os.path.basename(output_filename):
        ax.set_ylim(0, 0.16009476)   # y-range of the paper figure
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)


    protocol_legend_handles = [
        Patch(facecolor=protocol_palettes['Encoding'][len(p_local_values)//2], edgecolor='black', label='EE-T'),
        Patch(facecolor=protocol_palettes['Protocol_2G'][len(p_local_values)//2], edgecolor='black', label='EE-NC'),
    ]
    
    gray_palette = sns.color_palette("Greys_r", n_colors=len(p_local_values))
    local_error_legend_handles = [
        Patch(facecolor=gray_palette[i], edgecolor='black', label=f"{p_local_values[i]}")
        for i in range(len(p_local_values))
    ]
    
    contact_point_x = 0.2
    legend1 = ax.legend(handles=protocol_legend_handles, title="Protocol", 
                        loc='upper left', 
                        bbox_to_anchor=(contact_point_x, 1.0), 
                        frameon=True)
    legend2 = ax.legend(handles=local_error_legend_handles, title="Local Error",
                        loc='upper right',
                        bbox_to_anchor=(contact_point_x, 1.0), 
                        frameon=True)

    ax.add_artist(legend1)
    ax.grid(True, zorder=0)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.03)
    plt.show()


if __name__ == '__main__':
    plt.rcParams.update({
        "font.size": 25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.fontsize": 22,
        "axes.labelsize": 28,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "axes.titlesize": 28,
    })

    p_local_list = [0.0, 0.0005, 0.001, 0.002]
    p_transmission_list = [0.005, 0.025, 0.045, 0.065, 0.085, 0.105, 0.125, 0.145]
    
    for code_type in ["surface", "steane"]:
        full_data_df = prepare_data(p_local_list, p_transmission_list, code_type=code_type)
        
        create_comparison_bar_chart(
            df=full_data_df, 
            y_col='logical_error', 
            y_label='Logical Error Rate',
            title='Logical Error Rate Comparison', 
            output_filename= os.path.join(os.path.dirname(__file__),f'{code_type}_encoding_vs_2gen.pdf')
        )