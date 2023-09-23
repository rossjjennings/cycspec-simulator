import numpy as np
import numba as nb

from .polarization import coherence_to_stokes

@nb.njit
def fold_numba(phi, A, B, nbin):
    AA = np.zeros(nbin, dtype=np.float64)
    BB = np.zeros(nbin, dtype=np.float64)
    CR = np.zeros(nbin, dtype=np.float64)
    CI = np.zeros(nbin, dtype=np.float64)
    samples = np.zeros(nbin, dtype=np.int64)
    for i in range(phi.size):
        phase = phi[i] % 1
        phase_bin = np.int64(np.round(phase*nbin)) % nbin
        samples[phase_bin] += 1
        AA[phase_bin] += (A[i]*A[i].conjugate()).real
        BB[phase_bin] += (B[i]*B[i].conjugate()).real
        CR[phase_bin] += (A[i]*B[i].conjugate()).real
        CI[phase_bin] += (A[i]*B[i].conjugate()).imag
    AA /= samples
    BB /= samples
    CR /= samples
    CI /= samples
    return AA, BB, CR, CI

def fold(data, nbin, phase_predictor):
    phi = phase_predictor.phase(data.t)
    I = np.zeros((data.nchan, nbin))
    Q = np.zeros((data.nchan, nbin))
    U = np.zeros((data.nchan, nbin))
    V = np.zeros((data.nchan, nbin))
    for ichan in range(data.nchan):
        AA, BB, CR, CI = fold_numba(phi, data.A[ichan], data.B[ichan], nbin)
        I[ichan], Q[ichan], U[ichan], V[ichan] = (
            coherence_to_stokes(AA, BB, CR, CI, data.feed_poln)
        )
    return I, Q, U, V
