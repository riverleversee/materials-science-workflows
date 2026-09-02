"""
Plotting helpers for group-resolved normal-mode results JSON.

Loads ``final_analysismodes_results.json`` (or similar) and writes publication figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import itertools


minpress=['166','188','204']
minamb=['172','189','210']

mode_frequencies = {
    "179": 906,
    "200":1043,
    "217": 1319,
    "225": 1355,
    "210":1286,
    "189": 1001,
    "172": 825,
    "202": 1120,
    "188": 961,
    "166":806,
    "183":911,


    # Add more as needed
}


def plot_spread_single_panel(results_dict, mode_names_to_highlight, output_filename="spread_plot.png"):
    """
    Plots the 'spread' metric for all modes in a single panel.
    Highlights specified modes with dashed lines during initial plotting.
    Adds mode frequency info to legend using external dictionary.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      mode_names_to_highlight: List of mode names to highlight with dashed lines.
      output_filename: Name of the file to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    all_handles = []
    all_labels = []

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        spread_values = mode_results["spread"]
        color = next(color_cycle)

        linestyle = "--" if mode in mode_names_to_highlight else "-"
        sensitivity = "Insensitive" if mode in mode_names_to_highlight else "Sensitive"
        freq = mode_frequencies.get(mode, "N/A")

        label = f"Mode {freq} cm⁻¹ ({sensitivity})"

        line, = ax.plot(
            pressures,
            spread_values,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label
        )

        all_handles.append(line)
        all_labels.append(label)

    # Axis formatting
    ax.set_xlabel("Pressure (GPa)", fontsize=20)
    ax.set_ylabel("Normalized RMS Disp. Mag.", fontsize=20)
    ax.set_title("Pressure Sensitivity and Mode Localization", fontsize=22)
    ax.tick_params(axis='both', labelsize=16)

    # Legend outside the plot
   # fig.legend(
   #     handles=all_handles,
   #     labels=all_labels,
   #     loc="center right",
   #     bbox_to_anchor=(1.25, 0.5),
   #     fontsize=12
   # )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for legend
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved spread plot as '{output_filename}'.")





def plot_mag_single_panel(results_dict, mode_names_to_highlight, output_filename="spread_plot.png"):
    """
    Plots the 'spread' metric for all modes in a single panel.
    Highlights specified modes with dashed lines during initial plotting.
    Adds mode frequency info to legend using external dictionary.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      mode_names_to_highlight: List of mode names to highlight with dashed lines.
      output_filename: Name of the file to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    all_handles = []
    all_labels = []

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        spread_values = mode_results["total_magnitude"]
        color = next(color_cycle)

        linestyle = "--" if mode in mode_names_to_highlight else "-"
        sensitivity = "Insensitive" if mode in mode_names_to_highlight else "Sensitive"
        freq = mode_frequencies.get(mode, "N/A")

        label = f"Mode {freq} cm⁻¹ ({sensitivity})"

        line, = ax.plot(
            pressures,
            spread_values,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label
        )

        all_handles.append(line)
        all_labels.append(label)

    # Axis formatting
    ax.set_xlabel("Pressure (GPa)", fontsize=20)
    ax.set_ylabel("Displacement Magnitude", fontsize=20)
    ax.set_title("Pressure Sensitivity and Total Displacement Magnitude", fontsize=22)
    ax.tick_params(axis='both', labelsize=16)

    # Legend outside the plot
  #  fig.legend(
  #      handles=all_handles,
  #      labels=all_labels,
  #      loc="center right",
  #      bbox_to_anchor=(1.25, 0.5),
  #      fontsize=12
  #  )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for legend
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved spread plot as '{output_filename}'.")

def plot_mag_single_panel(results_dict, mode_names_to_highlight, output_filename="spread_plot.png"):
    """
    Plots the 'spread' metric for all modes in a single panel.
    Highlights specified modes with dashed lines during initial plotting.
    Adds mode frequency info to legend using external dictionary.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      mode_names_to_highlight: List of mode names to highlight with dashed lines.
      output_filename: Name of the file to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    all_handles = []
    all_labels = []

    # Compute global maximum magnitude across all modes and pressures
    global_max = max(
        np.max(mode_results["total_magnitude"])
        for mode_results in results_dict.values()
    )

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        spread_values = mode_results["total_magnitude"]
        normalized_values = np.array(spread_values) / global_max

        color = next(color_cycle)
        linestyle = "--" if mode in mode_names_to_highlight else "-"
        sensitivity = "Insensitive" if mode in mode_names_to_highlight else "Sensitive"
        freq = mode_frequencies.get(mode, "N/A")

        label = f"Mode {freq} cm⁻¹ ({sensitivity})"

        line, = ax.plot(
            pressures,
            normalized_values,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label
        )

        all_handles.append(line)
        all_labels.append(label)

    # Axis formatting
    ax.set_xlabel("Pressure (GPa)", fontsize=20)
    ax.set_ylabel("Normalized Disp. Magnitude", fontsize=20)
    ax.set_title("Pressure Sensitivity and Normalized Displacement Magnitude", fontsize=22)
    ax.tick_params(axis='both', labelsize=16)

    # Optional legend outside the plot
    # fig.legend(
    #     handles=all_handles,
    #     labels=all_labels,
    #     loc="center right",
    #     bbox_to_anchor=(1.25, 0.5),
    #     fontsize=12
    # )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for legend
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

def plot_combined_mode_analysis(results_dict, mode_names_to_highlight, output_filename="combined_modes_plot.png"):
    """
    Creates a 4-panel figure with each subplot showing results across all normal modes.
    Overlays specified modes with dashed yellow lines.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      mode_names_to_highlight: List of mode names (keys in results_dict) to highlight with yellow dashed lines.
      output_filename: Name of the file to save the final figure.

    Outputs:
      - A single 4-panel plot where each subplot contains lines for all normal modes, with a unified legend.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'right': 0.8})

    metrics = [
        "com_disp_mag",
        "normalized_distance_scaled_sum_all_atoms",
        "spread_unnorm",
        "distance_scaled_sum_all_atoms"
    ]
    
    #metrics = [
    #    "nitro_fraction_total",
    #    "nitro_fraction_ON",
    #    "furazan_fraction_total",
   #     "furazan_fraction_ON"
   # ]
    titles = [
        "com_disp_mag",
        "Normalized Distance-Scaled Sum (All Atoms)",
        "spread",
        "Unnormalized Distance Scaling"
    ]

  #  titles = [
  #      "nitro_fraction_total",
  #      "nitro_fraction_ON",
  #      "furazan_fraction_total",
  #      "furazan_fraction_ON"
  #  ]


    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}
    all_handles = []
    all_labels = []

    # Plot each mode with a unique color
    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        for ax, metric, title in zip(axs.flatten(), metrics, titles):
            line, = ax.plot(
                pressures,
                mode_results[metric],
                marker="o",
                linestyle="-",
                color=used_colors[mode],
                label=f"Mode {mode}"
            )
            if ax == axs[0, 0]:  # Collect legend items only once
                all_handles.append(line)
                all_labels.append(f"Mode {mode}")

    # Overlay highlighted modes in dashed yellow
    for mode in mode_names_to_highlight:
        if mode in results_dict:
            mode_results = results_dict[mode]
            pressures = mode_results["pressures"]

            for ax, metric, _ in zip(axs.flatten(), metrics, titles):
                line, = ax.plot(
                    pressures,
                    mode_results[metric],
                    linestyle="--",
                    color="yellow",
                    linewidth=2,
                    label=f"Highlighted {mode}"
                )
                if ax == axs[0, 0]:
                    all_handles.append(line)
                    all_labels.append(f"Highlighted {mode}")

    # Format subplots
    for ax, title in zip(axs.flatten(), titles):
        ax.set_xlabel("Pressure")
        ax.set_ylabel(title)
        ax.set_title(title + "")

    # Add a single legend to the right of all plots
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="center right",
        borderaxespad=1.0,
        fontsize="small"
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leave space on the right for legend
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved combined normal mode plot as '{output_filename}'.")



def plot_group_analysis(results_dict, mode_names_to_highlight, output_filename="combined_modes_plot_nolegend.png"):
    """
    Creates a 4-panel figure with each subplot showing results across all normal modes.
    Highlights specified modes with dashed lines. Removes legend and increases font sizes.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing processed results for each mode over pressure.
    mode_names_to_highlight : list
        List of mode names to highlight with dashed lines.
    output_filename : str
        Name of the file to save the final figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        "nitro_fraction_total",
        "nitro_fraction_ON",
        "furazan_fraction_total",
        "furazan_fraction_ON"
    ]
    titles = [
        "Total Nitro Group Fraction",
        "Nitro Group Fraction (O and N only)",
        "Total Furazan Ring Fraction",
        "Furazan Ring Fraction (O and N only)"
    ]

    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        linestyle = "--" if mode in mode_names_to_highlight else "-"

        for ax, metric, title in zip(axs.flatten(), metrics, titles):
            ax.plot(
                pressures,
                mode_results[metric],
                marker="o",
                linestyle=linestyle,
                color=used_colors[mode]
            )

    # Format subplots
    for ax, title in zip(axs.flatten(), titles):
        ax.set_xlabel("Pressure (GPa)", fontsize=20)
        ax.set_ylabel(title, fontsize=20)
        ax.set_title(title + "", fontsize=22)
        ax.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved combined normal mode plot without legend as '{output_filename}'.")




import matplotlib.pyplot as plt
import itertools

def plot_group_analysis(results_dict, mode_names_to_highlight, output_filename="combined_modes_plot_nolegend.png"):
    """
    Modified: Adds overall title, column labels, and simplified axis labels.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        "nitro_fraction_total",
        "nitro_fraction_ON",
        "furazan_fraction_total",
        "furazan_fraction_ON"
    ]
    ylabels = [
        "Nitro Group Fraction",
        "Nitro Group Fraction",
        "Furazan Ring Fraction",
        "Furazan Ring Fraction"
    ]

    column_labels = ["All Atoms", "O and N Only"]
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        linestyle = "--" if mode in mode_names_to_highlight else "-"

        for ax, metric in zip(axs.flatten(), metrics):
            ax.plot(
                pressures,
                mode_results[metric],
                marker="o",
                linestyle=linestyle,
                color=used_colors[mode]
            )

    # Format subplots
    for i, ax in enumerate(axs.flatten()):
        ax.set_xlabel("Pressure (GPa)", fontsize=20)
        ax.set_ylabel(ylabels[i], fontsize=20)
        ax.tick_params(axis='both', labelsize=16)

    # Add column labels
    for col in range(2):
        axs[0, col].annotate(
            column_labels[col],
            xy=(0.5, 1.08),
            xycoords='axes fraction',
            ha='center',
            fontsize=22
          
        )

    # Add overall title
    fig.suptitle("Nitro Group and Furazan Ring Character", fontsize=24,y=0.935)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved combined normal mode plot without legend as '{output_filename}'.")

def plot_combined_mode_analysis_rad(results_dict, mode_names_to_highlight, output_filename="combined_modes_plot.png"):
    """
    Creates a 4-panel figure with each subplot showing results across all normal modes.
    Overlays specified modes with dashed yellow lines.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      mode_names_to_highlight: List of mode names (keys in results_dict) to highlight with yellow dashed lines.
      output_filename: Name of the file to save the final figure.

    Outputs:
      - A single 4-panel plot where each subplot contains lines for all normal modes, with a unified legend.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'right': 0.8})

    metrics = [
        "normalized_radial_projection",
        "normalized_orthogonal_projection",
        "radial_projection",
        "orthogonal_projection"
    ]
    titles = [
        "Normalized Threshold Sum (All Atoms)",
        "Normalized Distance-Scaled Sum (All Atoms)",
        "Unnormalized Threshold",
        "Unnormalized Distance Scaling"
    ]
    titles = [
        "normalized_radial_projection",
        "normalized_orthogonal_projection",
        "radial_projection",
        "orthogonal_projection"
    ]


    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}
    all_handles = []
    all_labels = []

    # Plot each mode with a unique color
    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        for ax, metric, title in zip(axs.flatten(), metrics, titles):
            line, = ax.plot(
                pressures,
                mode_results[metric],
                marker="o",
                linestyle="-",
                color=used_colors[mode],
                label=f"Mode {mode}"
            )
            if ax == axs[0, 0]:  # Collect legend items only once
                all_handles.append(line)
                all_labels.append(f"Mode {mode}")

    # Overlay highlighted modes in dashed yellow
    for mode in mode_names_to_highlight:
        if mode in results_dict:
            mode_results = results_dict[mode]
            pressures = mode_results["pressures"]

            for ax, metric, _ in zip(axs.flatten(), metrics, titles):
                line, = ax.plot(
                    pressures,
                    mode_results[metric],
                    linestyle="--",
                    color="yellow",
                    linewidth=2,
                    label=f"Highlighted {mode}"
                )
                if ax == axs[0, 0]:
                    all_handles.append(line)
                    all_labels.append(f"Highlighted {mode}")

    # Format subplots
    for ax, title in zip(axs.flatten(), titles):
        ax.set_xlabel("Pressure")
        ax.set_ylabel(title)
        ax.set_title(title + "")
        ax.grid(True)

    # Add a single legend to the right of all plots
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="center right",
        borderaxespad=1.0,
        fontsize="small"
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leave space on the right for legend
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved combined normal mode plot as '{output_filename}'.")


def plot_atomwise_mode_analysis(results_dict, mode_names_to_highlight, output_filename="atomwise_analysis"):
    """
    Creates stacked 4-panel figures for each atomic type, showing vibrational trends across normal modes.
    Overlays specified modes with dashed yellow lines.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      mode_names_to_highlight: List of mode names to highlight with dashed yellow lines.
      output_prefix: Prefix for saving output files.

    Outputs:
      - Saves a stacked 4-panel figure for each atomic type, with normal modes plotted together.
    """
    # Identify atomic types
    atomic_types = set()
    for mode_results in results_dict.values():
        atomic_types.update(mode_results["threshold_sums_by_type"].keys())

    metrics = ["threshold_sums_by_type", "distance_scaled_sums_by_type"]
    titles = ["Threshold Sum", "Distance-Scaled Sum"]

    # Unique color mapping
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for atom_type in atomic_types:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot each mode with unique colors
        for mode, mode_results in results_dict.items():
            pressures = mode_results["pressures"]

            # Assign color if not already mapped
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            for ax, metric, title in zip(axs.flatten(), metrics, titles):
                if atom_type in mode_results[metric]:
                    ax.plot(pressures, mode_results[metric][atom_type],
                            marker="o",
                            linestyle="-",
                            color=used_colors[mode],
                            label=f"Mode {mode}")

        # Overlay highlighted modes in dashed yellow
        for mode in mode_names_to_highlight:
            if mode in results_dict:
                mode_results = results_dict[mode]
                pressures = mode_results["pressures"]

                for ax, metric, _ in zip(axs.flatten(), metrics, titles):
                    if atom_type in mode_results[metric]:
                        ax.plot(pressures, mode_results[metric][atom_type],
                                linestyle="--",
                                color="yellow",
                                linewidth=2,
                                label=f"Highlighted {mode}")

        # Format each subplot
        for ax, title in zip(axs.flatten(), titles):
            ax.set_xlabel("Pressure")
            ax.set_ylabel(title)
            ax.set_title(f"{title} for {atom_type}")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"{output_filename}_{atom_type}.png")
        plt.close()
        print(f"Saved atomwise plot for {atom_type} as '{output_filename}_{atom_type}.png'.")







def plot_atomwise_distance_scaled_analysis_with_total(results_dict, mode_names_to_highlight, output_filename="atomwise_distance_scaled"):
    """
    Creates stacked panels for each atomic type + a fourth panel showing the sum across all types.
    Overlays highlighted modes in dashed yellow.
    """

    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    # Identify atomic types
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["distance_scaled_sums_by_type"].keys()
        if atom_type.lower() != "all atoms"  # We'll compute it manually
    })

    panel_labels = atomic_types + ["All atoms (summed)"]
    num_panels = len(panel_labels)

    # Set up figure and color cycle
    fig, axs = plt.subplots(num_panels, 1, figsize=(10, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        ax = axs[idx]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if atom_type in data["distance_scaled_sums_by_type"]:
                ax.plot(pressures,
                        data["distance_scaled_sums_by_type"][atom_type],
                        marker="o",
                        linestyle="-",
                        color=used_colors[mode],
                        label=f"Mode {mode}")

        for mode in mode_names_to_highlight:
            if mode in results_dict:
                data = results_dict[mode]
                pressures = data["pressures"]
                if atom_type in data["distance_scaled_sums_by_type"]:
                    ax.plot(pressures,
                            data["distance_scaled_sums_by_type"][atom_type],
                            linestyle="--",
                            color="yellow",
                            linewidth=2,
                            label=f"Highlighted {mode}")

        ax.set_ylabel(f"{atom_type}")
        ax.grid(True)
        ax.legend(
          loc='center left', 
          bbox_to_anchor=(1.0, 0.5)
        )

    # Fourth panel: Sum across atomic types
    ax_total = axs[-1]
    for mode, data in results_dict.items():
        pressures = data["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        summed_values = []
        for i in range(len(pressures)):
            total = sum(
                data["distance_scaled_sums_by_type"].get(atom_type, [0]*len(pressures))[i]
                for atom_type in atomic_types
            )
            summed_values.append(total)

        ax_total.plot(pressures,
                      summed_values,
                      marker="o",
                      linestyle="-",
                      color=used_colors[mode],
                      label=f"Mode {mode}")

    for mode in mode_names_to_highlight:
        if mode in results_dict:
            data = results_dict[mode]
            pressures = data["pressures"]
            summed_values = []
            for i in range(len(pressures)):
                total = sum(
                    data["distance_scaled_sums_by_type"].get(atom_type, [0]*len(pressures))[i]
                    for atom_type in atomic_types
                )
                summed_values.append(total)

            ax_total.plot(pressures,
                          summed_values,
                          linestyle="--",
                          color="yellow",
                          linewidth=2,
                          label=f"Highlighted {mode}")

    ax_total.set_ylabel("All atoms (sum)")
    ax_total.set_xlabel("Pressure")
    ax_total.grid(True)
    ax_total.legend(
      loc='center left', 
      bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle("Distance-Scaled Vibrational Sum by Atomic Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_filename}_distance_scaled_by_type.png")
    plt.close()

    print(f"Saved stacked plot including summed panel as '{output_filename}.png'.")










def plot_norm_atomwise_distance_scaled_analysis_with_total(results_dict, mode_names_to_highlight, output_filename="atomwise_distance_scaled"):
    """
    Creates stacked panels for each atomic type + a fourth panel showing the sum across all types.
    Overlays highlighted modes in dashed yellow.
    """

    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    # Identify atomic types
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["normalized_distance_scaled_sums_by_type"].keys()
        if atom_type.lower() != "all atoms"  # We'll compute it manually
    })

    panel_labels = atomic_types + ["All atoms (summed)"]
    num_panels = len(panel_labels)

    # Set up figure and color cycle
    fig, axs = plt.subplots(num_panels, 1, figsize=(10, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        ax = axs[idx]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if atom_type in data["normalized_distance_scaled_sums_by_type"]:
                ax.plot(pressures,
                        data["normalized_distance_scaled_sums_by_type"][atom_type],
                        marker="o",
                        linestyle="-",
                        color=used_colors[mode],
                        label=f"Mode {mode}")

        for mode in mode_names_to_highlight:
            if mode in results_dict:
                data = results_dict[mode]
                pressures = data["pressures"]
                if atom_type in data["normalized_distance_scaled_sums_by_type"]:
                    ax.plot(pressures,
                            data["normalized_distance_scaled_sums_by_type"][atom_type],
                            linestyle="--",
                            color="yellow",
                            linewidth=2,
                            label=f"Highlighted {mode}")

        ax.set_ylabel(f"{atom_type}")
        ax.grid(True)
        ax.legend(
          loc='center left',
          bbox_to_anchor=(1.0, 0.5)
        )

    # Fourth panel: Sum across atomic types
    ax_total = axs[-1]
    for mode, data in results_dict.items():
        pressures = data["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        summed_values = []
        for i in range(len(pressures)):
            total = sum(
                data["normalized_distance_scaled_sums_by_type"].get(atom_type, [0]*len(pressures))[i]
                for atom_type in atomic_types
            )
            summed_values.append(total)

        ax_total.plot(pressures,
                      summed_values,
                      marker="o",
                      linestyle="-",
                      color=used_colors[mode],
                      label=f"Mode {mode}")

    for mode in mode_names_to_highlight:
        if mode in results_dict:
            data = results_dict[mode]
            pressures = data["pressures"]
            summed_values = []
            for i in range(len(pressures)):
                total = sum(
                    data["normalized_distance_scaled_sums_by_type"].get(atom_type, [0]*len(pressures))[i]
                    for atom_type in atomic_types
                )
                summed_values.append(total)

            ax_total.plot(pressures,
                          summed_values,
                          linestyle="--",
                          color="yellow",
                          linewidth=2,
                          label=f"Highlighted {mode}")

    ax_total.set_ylabel("All atoms (sum)")
    ax_total.set_xlabel("Pressure")
    ax_total.grid(True)
    ax_total.legend(
      loc='center left',
      bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle("Normalized Distance-Scaled Vibrational Sum by Atomic Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_filename}_normalized_distance_scaled_by_type.png")
    plt.close()

    print(f"Saved stacked plot including summed panel as '{output_filename}.png'.")






def plot_atomwise_NNdist_with_total(results_dict, mode_names_to_highlight, output_filename="atomwise_weighted_dist"):

    """
    Creates stacked panels for each atomic type + a fourth panel showing the sum across all types.
    Overlays highlighted modes in dashed yellow.
    """

    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    # Identify atomic types
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["norm_outmolec_bytype"].keys()
        if atom_type.lower() != "all atoms"  # We'll compute it manually
    })

    panel_labels = atomic_types + ["All atoms (weighted mean)"]
    num_panels = len(panel_labels)

    # Set up figure and color cycle
    fig, axs = plt.subplots(num_panels, 1, figsize=(10, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        ax = axs[idx]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if atom_type in data["norm_outmolec_bytype"]:
                ax.plot(pressures,
                        data["norm_outmolec_bytype"][atom_type],
                        marker="o",
                        linestyle="-",
                        color=used_colors[mode],
                        label=f"Mode {mode}")

        for mode in mode_names_to_highlight:
            if mode in results_dict:
                data = results_dict[mode]
                pressures = data["pressures"]
                if atom_type in data["norm_outmolec_bytype"]:
                    ax.plot(pressures,
                            data["norm_outmolec_bytype"][atom_type],
                            linestyle="--",
                            color="yellow",
                            linewidth=2,
                            label=f"Highlighted {mode}")

        ax.set_ylabel(f"{atom_type}")
        ax.grid(True)
        ax.legend(
          loc='center left',
          bbox_to_anchor=(1.0, 0.5)
        )

    # Fourth panel: Sum across atomic types
    ax_total = axs[-1]
    for mode, data in results_dict.items():
        pressures = data["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        summed_values = data["norm_minoutmolec"]

        ax_total.plot(pressures,
                      summed_values,
                      marker="o",
                      linestyle="-",
                      color=used_colors[mode],
                      label=f"Mode {mode}")

    for mode in mode_names_to_highlight:
        if mode in results_dict:
            data = results_dict[mode]
            pressures = data["pressures"]
            summed_values = data["norm_minoutmolec"]

            ax_total.plot(pressures,
                          summed_values,
                          linestyle="--",
                          color="yellow",
                          linewidth=2,
                          label=f"Highlighted {mode}")

    ax_total.set_ylabel("All atoms (sum)")
    ax_total.set_xlabel("Pressure")
    ax_total.grid(True)
    ax_total.legend(
      loc='center left',
      bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle("Mode Weighted NN Intermolec Distance by Atomic Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_filename}_NNinterdist_by_type.png")
    plt.close()

    print(f"Saved stacked plot including summed panel as '{output_filename}.png'.")

    """
    Creates stacked panels for each atomic type + a fourth panel showing the sum across all types.
    Overlays highlighted modes in dashed yellow.
    """


    # Identify atomic types
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["norm_inmolec_bytype"].keys()
        if atom_type.lower() != "all atoms"  # We'll compute it manually
    })

    panel_labels = atomic_types + ["All atoms (weighted mean)"]
    num_panels = len(panel_labels)

    # Set up figure and color cycle
    fig, axs = plt.subplots(num_panels, 1, figsize=(10, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        ax = axs[idx]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if atom_type in data["norm_inmolec_bytype"]:
                ax.plot(pressures,
                        data["norm_inmolec_bytype"][atom_type],
                        marker="o",
                        linestyle="-",
                        color=used_colors[mode],
                        label=f"Mode {mode}")

        for mode in mode_names_to_highlight:
            if mode in results_dict:
                data = results_dict[mode]
                pressures = data["pressures"]
                if atom_type in data["norm_inmolec_bytype"]:
                    ax.plot(pressures,
                            data["norm_inmolec_bytype"][atom_type],
                            linestyle="--",
                            color="yellow",
                            linewidth=2,
                            label=f"Highlighted {mode}")

        ax.set_ylabel(f"{atom_type}")
        ax.grid(True)
        ax.legend(
          loc='center left',
          bbox_to_anchor=(1.0, 0.5)
        )

    # Fourth panel: Sum across atomic types
    ax_total = axs[-1]
    for mode, data in results_dict.items():
        pressures = data["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        summed_values = data["norm_mininmolec"]

        ax_total.plot(pressures,
                      summed_values,
                      marker="o",
                      linestyle="-",
                      color=used_colors[mode],
                      label=f"Mode {mode}")

    for mode in mode_names_to_highlight:
        if mode in results_dict:
            data = results_dict[mode]
            pressures = data["pressures"]
            summed_values = data["norm_mininmolec"]

            ax_total.plot(pressures,
                          summed_values,
                          linestyle="--",
                          color="yellow",
                          linewidth=2,
                          label=f"Highlighted {mode}")

    ax_total.set_ylabel("All atoms (sum)")
    ax_total.set_xlabel("Pressure")
    ax_total.grid(True)
    ax_total.legend(
      loc='center left',
      bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle("Mode Weighted NN Intramolec Distance by Atomic Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_filename}_NNintradist_by_type.png")
    plt.close()

    print(f"Saved stacked plot including summed panel as '{output_filename}.png'.")


def plot_atomwise_NNdist_with_total_SI(results_dict, mode_names_to_highlight, output_filename="atomwise_weighted_dist_nolegend"):
    """
    Creates a 2-column grid of panels for each atomic type showing intermolecular and intramolecular NN distances.
    Highlights specified modes with dashed lines. Removes legend and increases font sizes.
    """
    # Identify atomic types from both intermolecular and intramolecular keys
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["norm_outmolec_bytype"].keys()
        if atom_type.lower() != "all atoms"
    })

    panel_labels = atomic_types + ["All atoms (weighted mean)"]
    num_panels = len(panel_labels)

    fig, axs = plt.subplots(num_panels, 2, figsize=(16, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        for col, key, title in zip(
            [0, 1],
            ["norm_outmolec_bytype", "norm_inmolec_bytype"],
            ["Intermolecular", "Intramolecular"]
        ):
            ax = axs[idx, col]

            for mode, data in results_dict.items():
                pressures = data["pressures"]
                if mode not in used_colors:
                    used_colors[mode] = next(color_cycle)

                if atom_type in data[key]:
                    linestyle = "--" if mode in mode_names_to_highlight else "-"
                    ax.plot(
                        pressures,
                        data[key][atom_type],
                        marker="o",
                        linestyle=linestyle,
                        color=used_colors[mode]
                    )

            ax.set_ylabel(f"{atom_type} Atoms", fontsize=20)
            ax.set_xlabel("Pressure (GPa)", fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            if idx == 0:
                ax.set_title(title, fontsize=22)

    # Final row: total values
    for col, key, label in zip(
        [0, 1],
        ["norm_minoutmolec", "norm_mininmolec"],
        ["Intermolecular", "Intramolecular"]
    ):
        ax = axs[-1, col]
        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            linestyle = "--" if mode in mode_names_to_highlight else "-"
            ax.plot(
                pressures,
                data[key],
                marker="o",
                linestyle=linestyle,
                color=used_colors[mode]
            )

        ax.set_ylabel("All atoms (sum)", fontsize=20)
        ax.set_xlabel("Pressure (GPa)", fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
       # ax.set_title(label, fontsize=16)

    fig.suptitle("Mode Weighted Nearest Neighbor Distances", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{output_filename}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved 2-column NN distance plot as '{output_filename}.png'.")



import matplotlib.pyplot as plt
import itertools

def plot_atomwise_NNdist_with_total_SI(results_dict, mode_names_to_highlight, output_filename="atomwise_weighted_dist_nolegend"):
    """
    Modified: Only plots oxygen and nitrogen panels in 2-column layout. Excludes carbon and total panels.
    """
    # Filter to only oxygen and nitrogen
    atomic_types = [atom for atom in ["O", "N"] if any(
        atom in mode_data["norm_outmolec_bytype"] for mode_data in results_dict.values()
    )]

    num_panels = len(atomic_types)
    fig, axs = plt.subplots(num_panels, 2, figsize=(16, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        for col, key, title in zip(
            [0, 1],
            ["norm_outmolec_bytype", "norm_inmolec_bytype"],
            ["Intermolecular", "Intramolecular"]
        ):
            ax = axs[idx, col]

            for mode, data in results_dict.items():
                pressures = data["pressures"]
                if mode not in used_colors:
                    used_colors[mode] = next(color_cycle)

                if atom_type in data[key]:
                    linestyle = "--" if mode in mode_names_to_highlight else "-"
                    ax.plot(
                        pressures,
                        data[key][atom_type],
                        marker="o",
                        linestyle=linestyle,
                        color=used_colors[mode]
                    )

            ax.set_ylabel(f"{atom_type} Atoms", fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel(f"Pressure (GPa)", fontsize=20)
            if idx == 0:
                ax.set_title(title, fontsize=22)

    fig.suptitle("Mode Weighted Nearest Neighbor Distances", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{output_filename}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved O/N-only NN distance plot as '{output_filename}.png'.")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools


def plot_atomwise_NNdist_with_total_SI(results_dict, mode_names_to_highlight, settings=None, output_filename="atomwise_weighted_dist_nolegend"):
    """
    Plots atom-wise distances with enhanced control over broken axis settings,
    maintaining a fixed visual size for the upper broken plot region.

    Args:
        results_dict (dict): Dictionary containing plotting data.
        mode_names_to_highlight (list): List of modes to highlight with dashed lines.
        settings (dict, optional): Custom settings for plot appearance.
            Example:
            {
                "C": {
                    "inter_ylim_lower": (3.0, 3.2),
                    "inter_ylim_upper": (3.2, 3.4),
                    "inter_yticks_lower": [3.0, 3.1, 3.2],
                    "inter_yticks_upper": [3.2, 3.3, 3.4],
                    "intra_ylim": (2.2, 2.8),
                    "intra_yticks": [2.2, 2.4, 2.6, 2.8]
                },
                "spacing_factor": 0.05,  # Space between upper and lower broken axes
                "vertical_spacing": 0.2,  # Space between each atom-wise row
                "split_ratio": 0.3      # Visual height ratio of the upper broken plot
            }
        output_filename (str): The filename for the saved plot.
    """
    atom_order = ["C", "O", "N"]
    atomic_types = [atom for atom in atom_order if any(
        atom in mode_data["norm_outmolec_bytype"] for mode_data in results_dict.values()
    )]

    default_settings = {
        "C": {
            "inter_ylim_lower": (2.65, 3.18),
            "inter_ylim_upper": (3.18, 3.34),
            "intra_ylim": (1.308, 1.33),
            # Custom tick marks for C intramolecular plot
            "intra_yticks": [1.31, 1.32, 1.33], 
        },
        "O": {
            "inter_ylim_lower": (2.4, 3.045),
            "inter_ylim_upper": (3.045, 3.085),
            "intra_ylim": (1.24, 1.35),
        },
        "N": {
            "inter_ylim_lower": (2.6, 3.05),
            "inter_ylim_upper": (3.05, 3.35),
            "intra_ylim": (1.27, 1.32),
        },
        "spacing_factor": 0.15,
        "vertical_spacing": 0.2,
        "split_ratio": 0.5
    }

    # Merge default settings with user-provided settings
    if settings:
        for atom_type, atom_settings in settings.items():
            if atom_type in default_settings:
                default_settings[atom_type].update(atom_settings)
            else:
                default_settings[atom_type] = atom_settings
        if "spacing_factor" in settings:
            default_settings["spacing_factor"] = settings["spacing_factor"]
        if "vertical_spacing" in settings:
            default_settings["vertical_spacing"] = settings["vertical_spacing"]
        if "split_ratio" in settings:
            default_settings["split_ratio"] = settings["split_ratio"]

    # Calculate height ratios for gridspec to control visual size
    height_ratios = []
    split_ratio = default_settings["split_ratio"]
    for i, atom_type in enumerate(atomic_types):
        height_ratios.extend([split_ratio, 1 - split_ratio])
        if i < len(atomic_types) - 1:
            height_ratios.append(default_settings["vertical_spacing"])

    num_rows_gridspec = len(height_ratios)
    num_atoms = len(atomic_types)
    fig = plt.figure(figsize=(16, 5 * num_atoms))
    gs = gridspec.GridSpec(num_rows_gridspec, 2, height_ratios=height_ratios, hspace=default_settings["spacing_factor"])

    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    gs_index = 0
    for i, atom_type in enumerate(atomic_types):
        atom_settings = default_settings[atom_type]
        inter_ylim_lower = atom_settings.get("inter_ylim_lower", (None, None))
        inter_ylim_upper = atom_settings.get("inter_ylim_upper", (None, None))
        inter_yticks_lower = atom_settings.get("inter_yticks_lower")
        inter_yticks_upper = atom_settings.get("inter_yticks_upper")
        intra_ylim = atom_settings.get("intra_ylim", (None, None))
        intra_yticks = atom_settings.get("intra_yticks")

        # Intermolecular: broken y-axis
        ax_top = fig.add_subplot(gs[gs_index, 0])
        ax_bot = fig.add_subplot(gs[gs_index + 1, 0], sharex=ax_top)

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)
            if atom_type in data["norm_outmolec_bytype"]:
                linestyle = "--" if mode in mode_names_to_highlight else "-"
                values = data["norm_outmolec_bytype"][atom_type]
                ax_top.plot(pressures, values, marker="o", linestyle=linestyle, color=used_colors[mode])
                ax_bot.plot(pressures, values, marker="o", linestyle=linestyle, color=used_colors[mode])

        ax_top.set_ylim(*inter_ylim_upper)
        ax_bot.set_ylim(*inter_ylim_lower)
        if inter_yticks_upper:
            ax_top.set_yticks(inter_yticks_upper)
        if inter_yticks_lower:
            ax_bot.set_yticks(inter_yticks_lower)

        ax_bot.set_xlabel("Pressure (GPa)", fontsize=20)
        ax_top.tick_params(labelbottom=False, labelsize=16)
        ax_bot.tick_params(labelsize=16)

        if i == 0:
            ax_top.set_title("Intermolecular", fontsize=22)

        # Diagonal break marks
        d = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        # Centered y-axis label using figure-level coordinates
        fig.text(
            0.075,
            (ax_top.get_position().y0 + ax_bot.get_position().y1) / 2,
            f"{atom_type}-X (Å)",
            va='center',
            ha='center',
            rotation='vertical',
            fontsize=20
        )

        # Intramolecular: spans both rows
        ax_intra = fig.add_subplot(gs[gs_index : gs_index + 2, 1])
        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)
            if atom_type in data["norm_inmolec_bytype"]:
                linestyle = "--" if mode in mode_names_to_highlight else "-"
                values = data["norm_inmolec_bytype"][atom_type]
                ax_intra.plot(pressures, values, marker="o", linestyle=linestyle, color=used_colors[mode])

        ax_intra.set_ylim(*intra_ylim)
        if intra_yticks:
            ax_intra.set_yticks(intra_yticks)

        # Shift the y-label for intramolecular plots
        ax_intra.yaxis.set_label_coords(-0.115, 0.5) 

        ax_intra.set_xlabel("Pressure (GPa)", fontsize=20)
        ax_intra.set_ylabel(f"{atom_type}-X (Å)", fontsize=20)
        ax_intra.tick_params(labelsize=16)
        if i == 0:
            ax_intra.set_title("Intramolecular", fontsize=22)

        gs_index += 2
        if i < len(atomic_types) - 1:
            gs_index += 1

    fig.suptitle("Mode Weighted Nearest Neighbor Distances", fontsize=22,y=0.94)
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    plt.savefig(f"{output_filename}.png", bbox_inches="tight")
    plt.close()

def plot_atomwise_magnitude_with_total(results_dict, mode_names_to_highlight, output_filename="atomwise_distance_scaled"):

    """
    Creates stacked panels for each atomic type + a fourth panel showing the sum across all types.
    Overlays highlighted modes in dashed yellow.
    """

    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    # Identify atomic types
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["magnitude_by_type"].keys()
        if atom_type.lower() != "all atoms"  # We'll compute it manually
    })

    panel_labels = atomic_types + ["All atoms (summed)"]
    num_panels = len(panel_labels)

    # Set up figure and color cycle
    fig, axs = plt.subplots(num_panels, 1, figsize=(10, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        ax = axs[idx]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if atom_type in data["magnitude_by_type"]:
                ax.plot(pressures,
                        data["magnitude_by_type"][atom_type],
                        marker="o",
                        linestyle="-",
                        color=used_colors[mode],
                        label=f"Mode {mode}")

        for mode in mode_names_to_highlight:
            if mode in results_dict:
                data = results_dict[mode]
                pressures = data["pressures"]
                if atom_type in data["magnitude_by_type"]:
                    ax.plot(pressures,
                            data["magnitude_by_type"][atom_type],
                            linestyle="--",
                            color="yellow",
                            linewidth=2,
                            label=f"Highlighted {mode}")

        ax.set_ylabel(f"{atom_type}")
        ax.grid(True)
        ax.legend(
          loc='center left',
          bbox_to_anchor=(1.0, 0.5)
        )

    # Fourth panel: Sum across atomic types
    ax_total = axs[-1]
    for mode, data in results_dict.items():
        pressures = data["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        summed_values = []
        for i in range(len(pressures)):
            total = sum(
                data["magnitude_by_type"].get(atom_type, [0]*len(pressures))[i]
                for atom_type in atomic_types
            )
            summed_values.append(total)

        ax_total.plot(pressures,
                      summed_values,
                      marker="o",
                      linestyle="-",
                      color=used_colors[mode],
                      label=f"Mode {mode}")

    for mode in mode_names_to_highlight:
        if mode in results_dict:
            data = results_dict[mode]
            pressures = data["pressures"]
            summed_values = []
            for i in range(len(pressures)):
                total = sum(
                    data["magnitude_by_type"].get(atom_type, [0]*len(pressures))[i]
                    for atom_type in atomic_types
                )
                summed_values.append(total)

            ax_total.plot(pressures,
                          summed_values,
                          linestyle="--",
                          color="yellow",
                          linewidth=2,
                          label=f"Highlighted {mode}")

    ax_total.set_ylabel("All atoms (sum)")
    ax_total.set_xlabel("Pressure")
    ax_total.grid(True)
    ax_total.legend(
      loc='center left',
      bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle("Mode Magnitude Sum by Atomic Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_filename}_magnitude_by_type.png")
    plt.close()

    print(f"Saved stacked plot including summed panel as '{output_filename}.png'.")

def plot_atomwise_normalized_magnitude_with_total(results_dict, mode_names_to_highlight, output_filename="atomwise_distance_normalized"):
    """
    Creates stacked panels for each atomic type + a fourth panel showing the sum across all types.
    Each atomic type's magnitude is normalized by the total magnitude across all types at each pressure.
    Highlighted modes are shown in dashed yellow.
    """

    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    # Identify atomic types
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["magnitude_by_type"].keys()
        if atom_type.lower() != "all atoms"
    })

    panel_labels = atomic_types + ["All atoms (summed)"]
    num_panels = len(panel_labels)

    fig, axs = plt.subplots(num_panels, 1, figsize=(10, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        ax = axs[idx]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if atom_type in data["magnitude_by_type"]:
                raw_values = data["magnitude_by_type"][atom_type]
                total_values = [
                    sum(data["magnitude_by_type"].get(at, [0]*len(pressures))[i] for at in atomic_types)
                    for i in range(len(pressures))
                ]
                normalized_values = [
                    raw / total if total != 0 else 0
                    for raw, total in zip(raw_values, total_values)
                ]

                ax.plot(pressures,
                        normalized_values,
                        marker="o",
                        linestyle="-",
                        color=used_colors[mode],
                        label=f"Mode {mode}")

        for mode in mode_names_to_highlight:
            if mode in results_dict:
                data = results_dict[mode]
                pressures = data["pressures"]
                if atom_type in data["magnitude_by_type"]:
                    raw_values = data["magnitude_by_type"][atom_type]
                    total_values = [
                        sum(data["magnitude_by_type"].get(at, [0]*len(pressures))[i] for at in atomic_types)
                        for i in range(len(pressures))
                    ]
                    normalized_values = [
                        raw / total if total != 0 else 0
                        for raw, total in zip(raw_values, total_values)
                    ]

                    ax.plot(pressures,
                            normalized_values,
                            linestyle="--",
                            color="yellow",
                            linewidth=2,
                            label=f"Highlighted {mode}")

        ax.set_ylabel(f"{atom_type} (normalized)")
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Fourth panel: Total magnitude (not normalized)
    ax_total = axs[-1]
    for mode, data in results_dict.items():
        pressures = data["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        summed_values = [
            sum(data["magnitude_by_type"].get(atom_type, [0]*len(pressures))[i] for atom_type in atomic_types)
            for i in range(len(pressures))
        ]

        ax_total.plot(pressures,
                      summed_values,
                      marker="o",
                      linestyle="-",
                      color=used_colors[mode],
                      label=f"Mode {mode}")

    for mode in mode_names_to_highlight:
        if mode in results_dict:
            data = results_dict[mode]
            pressures = data["pressures"]
            summed_values = [
                sum(data["magnitude_by_type"].get(atom_type, [0]*len(pressures))[i] for atom_type in atomic_types)
                for i in range(len(pressures))
            ]

            ax_total.plot(pressures,
                          summed_values,
                          linestyle="--",
                          color="yellow",
                          linewidth=2,
                          label=f"Highlighted {mode}")

    ax_total.set_ylabel("All atoms (sum)")
    ax_total.set_xlabel("Pressure")
    ax_total.grid(True)
    ax_total.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.suptitle("Normalized Mode Magnitude by Atomic Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_filename}_normalized_magnitude_by_type.png")
    plt.close()

    print(f"Saved normalized stacked plot as '{output_filename}_normalized_magnitude_by_type.png'.")


def plot_atomwise_axis_proj(results_dict, mode_names_to_highlight, output_filename="atomwise_distance_scaled"):

    """
    Creates stacked panels for each atomic type + a fourth panel showing the sum across all types.
    Overlays highlighted modes in dashed yellow.
    """

    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    # Identify atomic types
    atomic_types = sorted({
        atom_type
        for mode_data in results_dict.values()
        for atom_type in mode_data["unnormaxis_projection"].keys()
        if atom_type.lower() != "all atoms"  # We'll compute it manually
    })

    panel_labels = atomic_types + ["All Axes"]
    num_panels = len(panel_labels)

    # Set up figure and color cycle
    fig, axs = plt.subplots(num_panels, 1, figsize=(10, 4 * num_panels), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}

    for idx, atom_type in enumerate(atomic_types):
        ax = axs[idx]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if atom_type in data["unnormaxis_projection"]:
                ax.plot(pressures,
                        data["unnormaxis_projection"][atom_type],
                        marker="o",
                        linestyle="-",
                        color=used_colors[mode],
                        label=f"Mode {mode}")

        for mode in mode_names_to_highlight:
            if mode in results_dict:
                data = results_dict[mode]
                pressures = data["pressures"]
                if atom_type in data["unnormaxis_projection"]:
                    ax.plot(pressures,
                            data["unnormaxis_projection"][atom_type],
                            linestyle="--",
                            color="yellow",
                            linewidth=2,
                            label=f"Highlighted {mode}")

        ax.set_ylabel(f"{atom_type}")
        ax.grid(True)
        ax.legend(
          loc='center left',
          bbox_to_anchor=(1.0, 0.5)
        )

    # Fourth panel: Sum across atomic types
    ax_total = axs[-1]
    for mode, data in results_dict.items():
        pressures = data["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

        summed_values = []
        for i in range(len(pressures)):
            total = sum(
                data["unnormaxis_projection"].get(atom_type, [0]*len(pressures))[i]
                for atom_type in atomic_types
            )
            summed_values.append(total)

        ax_total.plot(pressures,
                      summed_values,
                      marker="o",
                      linestyle="-",
                      color=used_colors[mode],
                      label=f"Mode {mode}")

    for mode in mode_names_to_highlight:
        if mode in results_dict:
            data = results_dict[mode]
            pressures = data["pressures"]
            summed_values = []
            for i in range(len(pressures)):
                total = sum(
                    data["unnormaxis_projection"].get(atom_type, [0]*len(pressures))[i]
                    for atom_type in atomic_types
                )
                summed_values.append(total)

            ax_total.plot(pressures,
                          summed_values,
                          linestyle="--",
                          color="yellow",
                          linewidth=2,
                          label=f"Highlighted {mode}")

    ax_total.set_ylabel("All atoms (sum)")
    ax_total.set_xlabel("Pressure")
    ax_total.grid(True)
#    ax_total.legend(
 #     loc='center left',
 #     bbox_to_anchor=(1.0, 0.5)
 #   )

    fig.suptitle("Mode axis proj Sum by Atomic Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_filename}_unnormaxis_projection.png")
    plt.close()

    print(f"Saved stacked plot including summed panel as '{output_filename}.png'.")



def plot_normalized_axis_proj(results_dict, mode_names_to_highlight, output_filename="normalized_axis_projection"):
    """
    Plots normalized axis projections (x, y, z) for each mode in a 3-row, 1-column layout.
    Highlights specified modes with dashed lines and includes mode frequency info in the legend.
    """
    axes = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    used_colors = {}
    all_handles = []
    all_labels = []

    for row, axis in enumerate(axes):
        ax = axs[row]

        for mode, data in results_dict.items():
            pressures = data["pressures"]
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)

            if axis in data["axis_projection"]:
                linestyle = "--" if mode in mode_names_to_highlight else "-"
                sensitivity = "Insensitive" if mode in mode_names_to_highlight else "Sensitive"
                freq = mode_frequencies.get(mode, "N/A")
                label = f"Mode {freq} cm⁻¹ ({sensitivity})"

                line, = ax.plot(
                    pressures,
                    data["axis_projection"][axis],
                    marker="o",
                    linestyle=linestyle,
                    color=used_colors[mode],
                    label=label
                )

                if row == 0:  # Collect legend items only once
                    all_handles.append(line)
                    all_labels.append(label)
        label_map = {"x": "[1 0 0] axis", "y": "[0 1 0] axis", "z": "[0 0 1] axis"}
        ax.set_ylabel(label_map[axis], fontsize=20)

        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True)

    axs[-1].set_xlabel("Pressure (GPa)", fontsize=20)
    fig.suptitle("Normalized Mode Axis Projections", fontsize=22)

    # Unified legend outside the plot
#    fig.legend(
#        handles=all_handles,
#        labels=all_labels,
#        loc="center right",
#        bbox_to_anchor=(1.25, 0.5),
#        fontsize=12
#    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{output_filename}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved normalized axis projection plot as '{output_filename}.png'.")








def runallplots(output_prefix):

  with open("final_analysismodes_results.json", "r") as f:
      results_dict = json.load(f)
  
  

  plot_combined_mode_analysis(results_dict,minpress, output_filename=f"{output_prefix}_combined_modes_minpress.png")
  plot_combined_mode_analysis(results_dict,minamb, output_filename=f"{output_prefix}_combined_modes_minamb.png")


  plot_group_analysis(results_dict,minpress, output_filename=f"{output_prefix}_groupanalysis_minpress.png")
  plot_group_analysis(results_dict,minamb, output_filename=f"{output_prefix}_groupanalysis_minamb.png")

  plot_combined_mode_analysis_rad(results_dict,minpress, output_filename=f"{output_prefix}_combined_modes_minpress_rad.png")
  plot_combined_mode_analysis_rad(results_dict,minamb, output_filename=f"{output_prefix}_combined_modes_minamb_rad.png")


  plot_atomwise_distance_scaled_analysis_with_total(results_dict,minamb, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_disatance_scaled_minamb")
  plot_atomwise_distance_scaled_analysis_with_total(results_dict,minpress, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_disatance_scaled_minpress")
  

  plot_norm_atomwise_distance_scaled_analysis_with_total(results_dict,minamb, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_norm_disatance_scaled_minamb")
  plot_norm_atomwise_distance_scaled_analysis_with_total(results_dict,minpress, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_norm_disatance_scaled_minpress")


  plot_atomwise_magnitude_with_total(results_dict,minamb, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_magnitude_minamb")
  plot_atomwise_magnitude_with_total(results_dict,minpress, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_magnitude_minpress")
 


  plot_atomwise_normalized_magnitude_with_total(results_dict,minamb, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_normalized_magnitude_minamb")
  plot_atomwise_normalized_magnitude_with_total(results_dict,minpress, output_filename=f"{output_prefix}_combined_modes_atomwise_analysis_normalized_magnitude_minpress")

  plot_normalized_axis_proj(results_dict,minpress, output_filename=f"combined_modes_axis_proj_minpress")
  plot_normalized_axis_proj(results_dict,minamb, output_filename=f"combined_modes_axis_proj_minamb")        

  plot_atomwise_NNdist_with_total(results_dict,minpress, output_filename=f"{output_prefix}_NNdist_min_press")
  plot_atomwise_NNdist_with_total(results_dict,minamb, output_filename=f"{output_prefix}_NNdist_min_amb")

  plot_atomwise_NNdist_with_total_SI(results_dict,minpress, output_filename=f"{output_prefix}_NNdist_min_press_SI")
  plot_atomwise_NNdist_with_total_SI(results_dict,minamb, output_filename=f"{output_prefix}_NNdist_min_amb_SI")


  plot_spread_single_panel(results_dict, minpress, output_filename=f"spread_plot_minpress.png")  
  plot_spread_single_panel(results_dict, minamb, output_filename=f"spread_plot_minamb.png")


  plot_mag_single_panel(results_dict, minpress, output_filename=f"mag_plot_minpress.png")
  plot_mag_single_panel(results_dict, minamb, output_filename=f"mag_plot_minamb.png")

if __name__ == "__main__":
    runallplots("modesout")





