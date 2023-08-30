import numpy as np
import numba as nb

from .polarization import coherence_to_stokes

@nb.njit
def fold_numba(data, nbin, phase_predictor):
    AA = np.zeros(nbin, dtype=np.float64)
    BB = np.zeros(nbin, dtype=np.float64)
    CR = np.zeros(nbin, dtype=np.float64)
    CI = np.zeros(nbin, dtype=np.float64)
    samples = np.zeros(nbin, dtype=np.int64)
    for i in range(data.t.size):
        phase = phase_predictor.phase(data.t[i]) % 1
        phase_bin = np.int64(np.round(phase*nbin)) % nbin
        samples[phase_bin] += 1
        AA[phase_bin] += (data.A[i]*data.A[i].conjugate()).real
        BB[phase_bin] += (data.B[i]*data.B[i].conjugate()).real
        CR[phase_bin] += (data.A[i]*data.B[i].conjugate()).real
        CI[phase_bin] += (data.A[i]*data.B[i].conjugate()).imag
    AA /= samples
    BB /= samples
    CR /= samples
    CI /= samples
    I, Q, U, V = coherence_to_stokes(AA, BB, CR, CI, data.feed_poln)
    return I, Q, U, V
