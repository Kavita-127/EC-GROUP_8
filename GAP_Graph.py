# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np


def plot_grouped_gap_histogram(gap8, gap10, gap12):
    algos = ["BCGA", "RCGA", "PSO", "DE", "TLBO", "ABC"]

    # Distinct hatch patterns
    hatches = ['', '//', '\\\\', '||', '--', 'xx']

    # Shades of gray — dark enough to be visible
    gray_shades = [0.85, 0.70, 0.55, 0.40, 0.25, 0.10]
    colors = [plt.cm.Greys(s) for s in gray_shades]

    def pad(values):
        return values + [np.nan] * (6 - len(values))

    gap8  = pad(gap8)
    gap10 = pad(gap10)
    gap12 = pad(gap12)

    # Rows = datasets, Cols = algorithms
    data = np.array([gap8, gap10, gap12])

    n_groups = data.shape[0]   # 3 datasets
    n_algos  = data.shape[1]   # 6 algorithms

    bar_width     = 0.24
    group_spacing = 0.5

    indices = np.arange(n_groups) * (n_algos * bar_width + group_spacing)

    fig, ax = plt.subplots(figsize=(13, 7))

    for i in range(n_algos):
        x_pos = indices + i * bar_width

        bars = ax.bar(
            x_pos,
            data[:, i],
            width=bar_width,
            edgecolor='black',
            linewidth=0.7,
            color=colors[i],
            hatch=hatches[i],
            label=algos[i],
            zorder=3
        )

        # Annotate values above each bar
        for j, val in enumerate(data[:, i]):
            if not np.isnan(val):
                ax.text(
                    x_pos[j],
                    val + (max(np.nanmax(data) - np.nanmin(data), 1) * 0.008),
                    f"{val:.2f}",
                    ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold'
                )
            else:
                # Mark missing ABC replacement with N/A
                ax.text(
                    x_pos[j],
                    np.nanmin(data) * 0.995,
                    "N/A",
                    ha='center', va='bottom',
                    fontsize=7, color='gray', style='italic'
                )

    # X-axis — center labels under each group
    group_centers = indices + (n_algos * bar_width) / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        ["GAP-8\n(8 Agents, 48 Jobs)",
         "GAP-10\n(10 Agents, 40 Jobs)",
         "GAP-12\n(10 Agents, 60 Jobs)"],
        fontsize=10
    )

    # Y-axis — start just below minimum for better visual proportion
    y_min = np.nanmin(data)
    y_max = np.nanmax(data)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.10)

    ax.set_xlabel(
        "Dataset  (Instance 1 of each file)",
        fontsize=11, labelpad=10
    )
    ax.set_ylabel(
        "Average Best Fitness  (higher = better)",
        fontsize=11, labelpad=10
    )
    ax.set_title(
        "Algorithm Performance Comparison on GAP\n"
        "(Average Best Fitness over 20 Runs - Penalty Method)",
        fontsize=12, fontweight='bold', pad=15
    )

    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # Legend
    ax.legend(
        loc='upper right',
        framealpha=0.9,
        fontsize=9,
        title="Algorithm",
        title_fontsize=9
    )

    # Note about ABC
    fig.text(
        0.5, 0.01,
        "Note: ABC Replacement method not implemented "
        "(incompatible with neighbourhood-based search)",
        ha='center', fontsize=8, color='gray', style='italic'
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("gap_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__":
    #        ["BCGA",    "RCGA",    "PSO",     "DE",      "TLBO",    "ABC"]
    gap12 = [1424.90,  1445.40,   1432.40,   1408.30,   1447.45,   1420.30]
    gap10 = [945.85,    942.36,    953.30,    954.00,    940.00,    933.95]
    gap8  = [959.00,   1070.00,   1085.00,   1006.00,   1050.00,   1065.65]

    plot_grouped_gap_histogram(gap8, gap10, gap12)