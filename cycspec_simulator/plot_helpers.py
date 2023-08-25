import numpy as np

def symmetrize_limits(data, vmin=None, vmax=None):
    '''
    Produce symmetric limits for a set of data based on the data itself and
    (optionally) explicitly supplied upper or lower limits.
    '''
    datamin, datamax = np.nanmin(data), np.nanmax(data)
    lim = max(-datamin, datamax)
    if vmax is not None:
        lim = min(lim, vmax)
    if vmin is not None:
        lim = min(lim, -vmin)
    vmin, vmax = -lim, lim
    return vmin, vmax
