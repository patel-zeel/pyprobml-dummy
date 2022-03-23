import matplotlib.pyplot as plt

DEFAULT_WIDTH = 6.0
GOLDEN_MEAN = (5 ** 0.5 - 1.0) / 2.0  # Aesthetic ratio
DEFAULT_HEIGHT = DEFAULT_WIDTH * GOLDEN_MEAN
# SPLINE_COLOR = 'gray'


def latexify(n_figures=1, fig_width=DEFAULT_WIDTH, fig_height=DEFAULT_HEIGHT):

    if n_figures > 1:
        fig_width = fig_width / n_figures

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams["pdf.fonttype"] = 42

    # Font sizes
    SIZE_SMALL = 9
    # SIZE_MEDIUM = 14
    SIZE_LARGE = 24
    # https://stackoverflow.com/a/39566040
    plt.rc("font", size=SIZE_SMALL)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_SMALL)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_SMALL)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_SMALL)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SIZE_SMALL)  # legend fontsize
    plt.rc("figure", titlesize=SIZE_LARGE)  # fontsize of the figure title

    # latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    plt.rcParams["backend"] = "ps"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(fig_width, fig_height))


# latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
def format_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        # ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # for axis in [ax.xaxis, ax.yaxis]:
    #     axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax
