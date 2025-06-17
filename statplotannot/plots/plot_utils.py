import matplotlib.pyplot as plt
import matplotlib as mpl


def fix_legend(
    ax, frameon=False, ncols=1, loc="best", only_labels=False, fw="regular", fs=8
):
    leg = ax.legend(
        loc=loc,
        frameon=frameon,
        prop={"weight": fw, "size": fs},
        title=None,
        title_fontsize=8,
        ncols=ncols,
        labelcolor="linecolor" if only_labels else None,
        scatterpoints=0 if only_labels else None,
        handlelength=0 if only_labels else None,
        handletextpad=0 if only_labels else None,
        # bbox_to_anchor=(1.1, 1.1),
    )

    if only_labels:
        for item in leg.legend_handles:
            item.set_visible(False)


def xtick_format(ax, rotation=0):
    ax.tick_params(axis="x", rotation=rotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
