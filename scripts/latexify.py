import os
import matplotlib.pyplot as plt

DEFAULT_WIDTH = 6.0
GOLDEN_MEAN = (5 ** 0.5 - 1.0) / 2.0  # Aesthetic ratio
DEFAULT_HEIGHT = DEFAULT_WIDTH * GOLDEN_MEAN
# SPLINE_COLOR = 'gray'


def latexify(width_scale_factor=1, fig_width=None, fig_height=None):
    """
    width_scale_factor: float, with this factor the figure will be scaled
    fig_width: float, width of the figure in inches (if this is specified, width_scale_factor is ignored)
    fig_height: float, height of the figure in inches
    """
    if fig_width is None:
        fig_width = DEFAULT_WIDTH / width_scale_factor

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
    plt.rc("figure", titlesize=SIZE_SMALL)  # fontsize of the figure title

    # latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    plt.rcParams["backend"] = "ps"
    if not "TEST_MODE" in os.environ:
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(fig_width, fig_height))
