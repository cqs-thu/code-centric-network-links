import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Patch

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from cals.cal_logical_error import *

def get_cached_result(p_local, p_trans, cache_file_path):
    """
    (Corrected) Reads a result from the specified cache file, matching the user's format.
    """
    if not os.path.exists(cache_file_path) or os.path.getsize(cache_file_path) == 0:
        return None
    
    cache_df = pd.read_csv(cache_file_path)
    
    res = cache_df[(cache_df['p_local'] == p_local) & (cache_df['p_transmission'] == p_trans)]
    
    if not res.empty:
        success_col_name = 'distill_success_rate'
        error_col_name = 'distill_logical_error'
        
        if success_col_name in res.columns and error_col_name in res.columns:
            return res.iloc[0][success_col_name], res.iloc[0][error_col_name]
    return None

def add_result_to_cache(p_local, p_trans, success_rate, logical_error, cache_file_path):
    """
    (Corrected) Adds a new result to the cache file, matching the user's format.
    """
    columns = ['p_local', 'p_transmission', 'distill_success_rate', 'distill_logical_error']
    new_data = pd.DataFrame([{
        'p_local': p_local, 'p_transmission': p_trans,
        'distill_success_rate': success_rate, 'distill_logical_error': logical_error
    }])
    
    if os.path.exists(cache_file_path) and os.path.getsize(cache_file_path) > 0:
        new_data.to_csv(cache_file_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(cache_file_path, mode='w', header=True, index=False)


def _two_basis(run):
    # acceptance does not depend on the readout basis; logical error = average over the Z and X bases
    acc_z, ler_z = run('Z')
    acc_x, ler_x = run('X')
    return (acc_z + acc_x) / 2, (ler_z + ler_x) / 2


def _basis_avg(result):
    acceptance, zz, xx = result
    return acceptance, (zz + xx) / 2


def prepare_data(p_local_list, p_transmission_list):
    """
    (Corrected) Runs calculations or loads from cache and returns a consolidated DataFrame.
    """
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    protocols_to_run = {
        # logical error rate = average of the Z- and X-basis failure probabilities, with noiseless final readout:
        # repetition code through its ZZ- and XX-parity errors, surface and Steane codes through Z- and X-basis runs (_two_basis)
        'Distill': {'cache': os.path.join(results_dir, 'distillation_basis_avg_cache.csv'),
                    'calc': lambda pl, pt: _basis_avg(cal_DEJMPS_detection_per_basis(depth=3, p_local=pl, p_trans=pt, shots=10_000_000))},
        'Surface': {'cache': os.path.join(results_dir, 'surface_two_basis_cache.csv'),
                    'calc': lambda pl, pt: _two_basis(lambda b: cal_surface_detection(d=3, num_layer=0, p_local=pl, p_trans=pt, shots=10_000_000, protocol_type="distillation", min_failures=100, basis=b))},
        'Steane':  {'cache': os.path.join(results_dir, 'steane_two_basis_cache.csv'),
                    'calc': lambda pl, pt: _two_basis(lambda b: cal_steane_detection(p_local=pl, p_trans=pt, shots=10_000_000, protocol_type=1, zx_detector=True, min_failures=100, basis=b))}
    }

    all_results = []
    for p_local in p_local_list:
        for p_trans in p_transmission_list:
            for name, funcs in protocols_to_run.items():
                cached_res = get_cached_result(p_local, p_trans, funcs['cache'])
                if cached_res:
                    print(f"Using cache for {name}: pl={p_local}, pt={p_trans}")
                    success_rate, logical_error = cached_res
                else:
                    print(f"Calculating for {name}: pl={p_local}, pt={p_trans}")
                    success_rate, logical_error = funcs['calc'](p_local, p_trans)
                    add_result_to_cache(p_local, p_trans, success_rate, logical_error, funcs['cache'])
                all_results.append({'p_local': p_local, 'p_transmission': p_trans, 'protocol': name,
                                    'success_rate': success_rate, 'logical_error': logical_error})
    return pd.DataFrame(all_results)


def create_comparison_bar_chart(df, y_col, y_label, title, output_filename, value_is_inverse=False, ylim=None):
    """
    (UPDATED) Creates and saves a layered, grouped bar chart.
    The new 'value_is_inverse' flag handles plots like Success Rate.
    """
    p_local_values = sorted(df['p_local'].unique())
    p_transmit_values = sorted(df['p_transmission'].unique())
    protocols = ['Distill', 'Surface', 'Steane']

    protocol_palettes = {
        'Distill': sns.color_palette("Greens_r", n_colors=len(p_local_values)),
        'Surface': sns.color_palette("Reds_r", n_colors=len(p_local_values)),
        'Steane': sns.color_palette("Blues_r", n_colors=len(p_local_values)),
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(p_transmit_values))
    xtick_labels = [f"{pt:.3f}" for pt in p_transmit_values]
    bar_width = 0.25

    # Determine loop order based on the plot type
    if value_is_inverse:
        # For Success Rate: plot tallest bars (low p_local) to shortest (high p_local)
        loop_indices = range(len(p_local_values)) 
    else:
        # For Logical Error: plot tallest bars (high p_local) to shortest (low p_local)
        loop_indices = range(len(p_local_values) - 1, -1, -1)

    for i, protocol in enumerate(protocols):
        bar_positions = x + (i - 1) * bar_width
        
        # Use the determined loop order to draw layers correctly
        for j in loop_indices:
            p_local = p_local_values[j]
            y_values = df[
                (df['protocol'] == protocol) & (df['p_local'] == p_local)
            ].sort_values('p_transmission')[y_col].values
            color = protocol_palettes[protocol][j]
            ax.bar(bar_positions, y_values, width=bar_width, color=color, edgecolor='black', zorder=2)

    ax.set_xlabel("Transmission Error Rate", fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    if title == 'Logical Error Rate Comparison':
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)

    code_legend_handles = [
        Patch(facecolor=protocol_palettes['Distill'][len(p_local_values)//2], edgecolor='black', label='Repetition Code'),
        Patch(facecolor=protocol_palettes['Surface'][len(p_local_values)//2], edgecolor='black', label='Surface Code'),
        Patch(facecolor=protocol_palettes['Steane'][len(p_local_values)//2], edgecolor='black', label='Steane Code'),
    ]
    

    gray_palette = sns.color_palette("Greys_r", n_colors=len(p_local_values))
    local_error_legend_handles = [
        Patch(facecolor=gray_palette[i], edgecolor='black', label=f"{p_local_values[i]}")
        for i in range(len(p_local_values))
    ]
    # the two legends are drawn as two rows above the axes ("Code:" on top, "Local Error:" below),
    # so that no legend box covers a bar
    kw = dict(frameon=True, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2)
    legend_local = ax.legend(handles=[Patch(alpha=0, label='Local Error:')] + local_error_legend_handles, ncol=5,
                             loc='lower center', bbox_to_anchor=(0.5, 1.0), **kw)
    fig.canvas.draw()
    row_h = legend_local.get_window_extent().transformed(ax.transAxes.inverted()).height
    legend_code = ax.legend(handles=[Patch(alpha=0, label='Code:')] + code_legend_handles, ncol=4,
                            loc='lower center', bbox_to_anchor=(0.5, 1.0 + row_h), **kw)
    ax.add_artist(legend_local)
    ax.grid(True, zorder=0)

    plt.tight_layout()
    # re-stack the two legend rows after tight_layout has changed the axes height
    for _ in range(3):
        fig.canvas.draw()
        row_h = legend_local.get_window_extent().transformed(ax.transAxes.inverted()).height
        legend_code.set_bbox_to_anchor((0.5, 1.0 + row_h), transform=ax.transAxes)
        plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.03)
    plt.show()


if __name__ == '__main__':
    plt.rcParams.update({
        "font.size": 25,
        "legend.fontsize": 20,
        "axes.labelsize": 28,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "axes.titlesize": 28,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    p_local_list = [0.0001, 0.001, 0.005, 0.01] 
    p_transmission_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]

    full_data_df = prepare_data(p_local_list, p_transmission_list)

    # Create the Logical Error Rate plot (normal behavior)
    create_comparison_bar_chart(
        df=full_data_df, y_col='logical_error', y_label='Logical Error Rate',
        title='Logical Error Rate Comparison', output_filename=os.path.join(os.path.dirname(__file__), 'logical_error_rate_comparison.pdf'),
        value_is_inverse=False, ylim=(1.138e-7, 0.2749)
    )

    # Create the Success Rate plot (inverse behavior)
    create_comparison_bar_chart(
        df=full_data_df, y_col='success_rate', y_label='Acceptance Probability',
        title='Success Rate Comparison', output_filename=os.path.join(os.path.dirname(__file__), 'success_rate_comparison.pdf'),
        value_is_inverse=True, ylim=(0, 0.9903)
    )