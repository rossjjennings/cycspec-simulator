import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits

class TemplateProfile:
    def __init__(self, I, Q=None, U=None, V=None):
        """
        Create a new template profile from I, Q, U, and V arrays.
        If one of Q, U, or V is present, all must be, and all must have the same shape as I.
        """
        self.I = I
        if Q is not None or U is not None or V is not None:
            if Q is None or U is None or V is None:
                raise ValueError("Supply either all Stokes parameters (I, Q, U, V) or I only.")
            if Q.shape != I.shape or U.shape != I.shape or V.shape != I.shape:
                raise ValueError(
                    "Shapes of arrays do not match: "
                    f"I.shape = {I.shape}, Q.shape = {Q.shape}, U.shape = {U.shape}, V.shape = {V.shape}"
                )
            self.Q = Q
            self.U = U
            self.V = V
            self.full_stokes = True
        else:
            self.full_stokes = False

    @classmethod
    def from_file(cls, filename):
        """
        Create a new template profile from a PSRFITS file.
        The data in the file should be fully scrunched in frequency and time
        (i.e., one subintegration, one channel).
        """
        hdul = fits.open(filename)
        data = hdul['SUBINT'].data['DATA']

        if (nsub := data.shape[0]) != 1:
            raise ValueError(f"Template should have 1 subintegration (found {nsub}).")
        if (nchan := data.shape[2]) != 1:
            raise ValueError(f"Template should have 1 channel (found {nchan}).")

        npol = data.shape[1]
        newshape = (1, npol, 1, 1)
        scale = hdul['SUBINT'].data['DAT_SCL'].reshape(newshape)
        offset = hdul['SUBINT'].data['DAT_OFFS'].reshape(newshape)
        profile = data*scale + offset

        pol_type = hdul['SUBINT'].header['POL_TYPE'].upper()
        feed_poln = hdul['PRIMARY'].header['FD_POLN'].upper()
        if pol_type in ['AA+BB', 'INTEN']:
            # Total intensity data
            I = profile.flatten()
            return TemplateProfile(I)
        elif pol_type == 'IQUV':
            # Full Stokes data
            I, Q, U, V = profile.reshape(4, -1)
            return TemplateProfile(I, Q, U, V)
        elif pol_type == 'AABBCRCI':
            # Coherence data - convert to Stokes
            if feed_poln == "LIN":
                # Linearly polarized feed
                XX, YY, CR, CI = profile.reshape(4, -1)
                I = XX + YY
                Q = XX - YY
                U = 2*CR
                V = 2*CI
            elif feed_poln == "CIRC":
                # Circularly polarized feed
                LL, RR, CR, CI = profile.reshape(4, -1)
                I = LL + RR
                Q = 2*CR
                U = 2*CI
                V = LL - RR
            else:
                raise ValueError(f"Unrecognized feed polarization '{feed_poln}'.")
            return TemplateProfile(I, Q, U, V)
        else:
            raise ValueError(f"Unrecognized polarization type '{pol_type}'.")
