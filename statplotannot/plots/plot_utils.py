import matplotlib.pyplot as plt


def fix_legend(ax, frameon=False, loc="best", only_labels=False):
    leg = ax.legend(
        loc=loc,
        frameon=frameon,
        fontsize=8,
        title=None,
        title_fontsize=8,
        # bbox_to_anchor=(1.1, 1.1),
    )
    if only_labels:
        ax.legend(
            labelcolor="linecolor",
            scatterpoints=0,
            handlelength=0,
            handletextpad=0,
        )
        for item in leg.legend_handles:
            item.set_visible(False)
