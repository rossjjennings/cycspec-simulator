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

def phase_intensity_colorbar(
    fig,
    ax,
    cax=None,
    ampmax=1.,
    phasecmap=cm.phase,
):
    """
    Create a colorbar for a pcolor-style plot that uses domain coloring to
    represent a complex-valued function of two real variables. Based on
    examples from the "screens" package by Marten van Kerkwijk and
    Rik van Lieshout (github.com/mhvk/screens).

    Parameters
    ----------
    fig: Figure object in which to place the colorbar.
    ax: Axes object containing the plot described by the colorbar.
    cax: Axes object in which to place the colorbar. If not specified,
        automatic placement is attempted.
    ampmax: Maximum value of amplitude, used to determine the correct
        scale for the vertical axis of the colorbar.
    phasecmap: Colormap used to determine the color for each phase.
        Defaults to the perceptually uniform cmocean "phase" colormap.
    """
    cbar = fig.colorbar(mpl.cm.ScalarMappable(), ax=ax, aspect=7.5)
    cbar_pos = fig.axes[-1].get_position()
    cbar.remove()

    nph = 36
    nalpha = 256
    phases = np.linspace(-np.pi, np.pi, nph, endpoint=False) + np.pi/nph
    alphas = np.linspace(0., 1., nalpha, endpoint=False) + 0.5/nalpha
    phasegrid, alphagrid = np.meshgrid(phases, alphas)

    if cax is None:
        cax = fig.add_axes(cbar_pos)
    cax.imshow(phasegrid, alpha=alphagrid,
               origin='lower', aspect='auto', interpolation='none',
               cmap=phasecmap, extent=[-np.pi, np.pi, 0., ampmax])
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    cax.set_xticks([-np.pi, 0., np.pi])
    cax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    cax.set_xlabel('phase (rad)')
    return cax
