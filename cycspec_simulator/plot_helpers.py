import numpy as np
import matplotlib as mpl
from cmocean import cm

def symmetrize_limits(data, vmin=None, vmax=None):
    """
    Produce symmetric limits for a set of data based on the data itself and
    (optionally) explicitly supplied upper or lower limits.
    """
    datamin, datamax = np.nanmin(data), np.nanmax(data)
    lim = max(-datamin, datamax)
    if vmax is not None:
        lim = min(lim, vmax)
    if vmin is not None:
        lim = min(lim, -vmin)
    vmin, vmax = -lim, lim
    return vmin, vmax

def complex_colorbar(
    ax,
    cax=None,
    vmin=0.,
    vmax=1.,
    gamma=1.,
    phasecmap=cm.phase,
    nph = 36,
    nalpha = 256,
):
    """
    Create a colorbar for a pcolor-style plot that uses domain coloring to
    represent a complex-valued function of two real variables. Based on
    examples from the "screens" package by Marten van Kerkwijk and
    Rik van Lieshout (github.com/mhvk/screens).

    Parameters
    ----------
    ax: Axes object containing the plot described by the colorbar.
    cax: Axes object in which to place the colorbar. If not specified,
        automatic placement is attempted.
    vmin: Minimum amplitude value to show on colorbar.
    vmax: Maximum amplitude value to show on colorbar.
    gamma: Power-law exponent used for gamma correction.
    phasecmap: Colormap used to determine the color for each phase.
        Defaults to the perceptually uniform cmocean "phase" colormap.
    nph: Number of phase points to use in colorbar
    nalpha: Number of amplitude points to use in colorbar

    Returns
    -------
    cax: Axes on which the colorbar was drawn.
    """
    fig = ax.figure
    cbar = fig.colorbar(mpl.cm.ScalarMappable(), ax=ax, aspect=7.5)
    cbar_pos = fig.axes[-1].get_position()
    cbar.remove()

    alpha_min = (vmin/vmax)**gamma
    phases = np.linspace(-np.pi, np.pi, nph, endpoint=False) + np.pi/nph
    alphas = np.linspace(alpha_min, 1., nalpha, endpoint=False) + 0.5/nalpha
    phasegrid, alphagrid = np.meshgrid(phases, alphas)

    if cax is None:
        cax = fig.add_axes(cbar_pos)

    def forward(x):
        return x**gamma

    def reverse(x):
        return x**(1/gamma)

    cax.set_yscale('function', functions=(forward, reverse))
    cax.set_ylim(vmin, vmax)
    cax.pcolormesh(
        phases,
        vmax*alphas**(1/gamma),
        phasegrid,
        alpha=alphagrid,
        cmap=phasecmap,
    )
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    cax.set_xticks([-np.pi, 0., np.pi])
    cax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    cax.set_xlabel('phase (rad)')
    return cax
