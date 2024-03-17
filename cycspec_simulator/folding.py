import numpy as np
import numba as nb

from .polarization import coherence_to_stokes

@nb.njit
def fold_numba(phi, A, B, nbin):
    AA = np.zeros(nbin, dtype=A.real.dtype)
    BB = np.zeros(nbin, dtype=A.real.dtype)
    CR = np.zeros(nbin, dtype=A.real.dtype)
    CI = np.zeros(nbin, dtype=A.real.dtype)
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

@nb.njit(parallel=True)
def fold_channelized(phi, A, B, nbin):
    nchan = A.shape[0]
    AA = np.zeros((nchan, nbin), dtype=A.real.dtype)
    BB = np.zeros((nchan, nbin), dtype=A.real.dtype)
    CR = np.zeros((nchan, nbin), dtype=A.real.dtype)
    CI = np.zeros((nchan, nbin), dtype=A.real.dtype)
    samples = np.zeros((nchan, nbin), dtype=np.int64)
    for ichan in nb.prange(nchan):
        for i in range(phi.size):
            phase = phi[i] % 1
            phase_bin = np.int64(np.round(phase*nbin)) % nbin
            samples[ichan, phase_bin] += 1
            AA[ichan, phase_bin] += (A[ichan, i]*A[ichan, i].conjugate()).real
            BB[ichan, phase_bin] += (B[ichan, i]*B[ichan, i].conjugate()).real
            CR[ichan, phase_bin] += (A[ichan, i]*B[ichan, i].conjugate()).real
            CI[ichan, phase_bin] += (A[ichan, i]*B[ichan, i].conjugate()).imag
    AA /= samples
    BB /= samples
    CR /= samples
    CI /= samples
    return AA, BB, CR, CI
