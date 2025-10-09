import numpy as np
import numba as nb
import dask
import dask.array as da
from scipy import signal, fft
from loguru import logger

from .baseband import BasebandData, DelayedRNG, get_time_axis
from .interpolation import lerp
from .time import Time
from .cycspec import PeriodicSpectrum, cycfold_cpu
from .folding import fold_channelized
from .polarization import coherence_to_stokes
from .gpu import have_cuda
if have_cuda:
    from .cycspec_gpu import cycfold_gpu

def pfb(x, nchan, ntap, window="hamming", fs=1.0):
    """
    Channelize a time series using a polyphase filterbank

    Parameters
    ----------
    x : ndarray
       The input time series.
    nchan : int
       The number of channels to form.
    ntap : int
       The number of PFB taps to use.
    window : str
       The windowing function to use for the PFB coefficients.
    fs : float
       The sampling frequency of the input data.

    Returns
    -------
    x_pfb : ndarray
       The channelized data

    Notes
    -----
    If the input data are real-valued then only positive frequencies
    are returned.
    """
    h = signal.firwin(ntap*nchan,cutoff=1.0/nchan, window="rectangular")
    h *= signal.get_window(window, ntap*nchan)
    ns_chan = x.shape[0]//nchan
    x = x[:ns_chan*nchan].reshape((ns_chan, nchan))
    h = h.reshape((ntap, nchan))
    xs = np.zeros((ns_chan-ntap+1, nchan), dtype=x.dtype)
    # This correlate is clever, but for very long time series it's the problem,
    # since it can't be trivially parallelized across blocks
    # (it sticks everything into a very long FFT).
    for ichan in range(nchan):
        xs[:,ichan] = signal.correlate(x[:,ichan], h[:,ichan], mode="valid")
    # need scipy fft to avoid dtype promotion (really??P)
    xpfb = fft.fft(xs, nchan, axis=1)
    xpfb *= np.sqrt(nchan)
    xpfb = fft.fftshift(xpfb.T, axes=0)

    return xpfb

def channelize(data, nchan, ntap=24, window="hamming", rechunk=True):
    """
    Channelize a BasebandData object using a polyphase filterbank.

    Parameters
    ----------
    data: BasebandData object
    nchan: Number of channels into which to split the data
    ntap: Number of polyphase fiterbank taps
    window: Window function used for polyphase filterbank
            (string interpreted by `scipy.signal.get_window()`)
    rechunk: Whether to re-chunk the input arrays so that the
             chunk size is a multiple of `nchan`. Without this,
             some samples will be skipped at block boundaries,
             leading to a slight drift.

    Returns
    -------
    channelized_data: ChannelizedData object
    """
    freqs = fft.fftshift(fft.fftfreq(nchan, d=1/data.bandwidth))
    freqs += data.obsfreq
    start_time = data.t[nchan*ntap//2]

    h = signal.firwin(ntap*nchan,cutoff=1.0/nchan, window="rectangular")
    h *= signal.get_window(window, ntap*nchan)
    h = h.reshape((ntap, nchan))
    ns_chan = data.A.shape[0]//nchan

    def process_block(block):
        out = np.zeros((ns_chan-ntap+1, nchan), dtype=block.dtype)
        for ichan in range(nchan):
            out[:,ichan] = signal.correlate(block[:,ichan], h[:,ichan], mode="valid")
        out = np.fft.fft(out, axis=1)

    if data.delayed:
        if rechunk and any(chunk % nchan for chunk in data.chunks[0]):
            new_chunk_size = int(np.ceil(max(data.A.chunks[0])/nchan))*nchan
            A = data.A.rechunk(new_chunk_size)
            B = data.B.rechunk(new_chunk_size)
        else:
            A = data.A
            B = data.B
        A = A[:ns_chan*nchan].reshape((ns_chan, nchan))
        B = B[:ns_chan*nchan].reshape((ns_chan, nchan))

        A = da.map_overlap(
            process_block, A, depth={0: (nchan*(ntap-1), 0)}, dtype=data.A.dtype,
        )
        B = da.map_overlap(
            process_block, B, depth={0: (nchan*(ntap-1), 0)}, dtype=data.A.dtype,
        )
        start_time = start_time.compute()
    else:
        A = process_block(data.A)
        B = process_block(data.B)
    return ChannelizedData(
        A.T, B.T,
        start_time=start_time,
        feed_poln=data.feed_poln,
        chan_bw=data.bandwidth/nchan, freqs=freqs,
    )

def channelize_old(data, nchan, ntap=24, window="hamming", rechunk=True):
    """
    Channelize a BasebandData object using a polyphase filterbank.

    Parameters
    ----------
    data: BasebandData object
    nchan: Number of channels into which to split the data
    ntap: Number of polyphase fiterbank taps
    window: Window function used for polyphase filterbank
            (string interpreted by `scipy.signal.get_window()`)
    rechunk: Whether to re-chunk the input arrays so that the
             chunk size is a multiple of `nchan`. Without this,
             some samples will be skipped at block boundaries,
             leading to a slight drift.

    Returns
    -------
    channelized_data: ChannelizedData object
    """
    freqs = fft.fftshift(fft.fftfreq(nchan, d=1/data.bandwidth))
    freqs += data.obsfreq
    start_time = data.t[nchan*ntap//2] # It works, but why???
    if data.delayed:
        if rechunk and any(chunk % nchan for chunk in data.chunks[0]):
            new_chunk_size = int(np.ceil(max(data.chunks[0])/nchan))*nchan
            data = data.rechunk(new_chunk_size)
        chunks = list(chunk//nchan for chunk in data.chunks[0])
        chunks[0] -= ntap - 1
        chunks = ((nchan,), tuple(chunks))
        A_pfb = da.map_overlap(
            pfb, data.A, nchan=nchan, ntap=ntap, window=window, fs=data.bandwidth,
            depth={0: (nchan*(ntap-1), 0)}, boundary=None, dtype=data.A.dtype,
            chunks=chunks
        )
        B_pfb = da.map_overlap(
            pfb, data.B, nchan=nchan, ntap=ntap, window=window, fs=data.bandwidth,
            depth={0: (nchan*(ntap-1), 0)}, boundary=None, dtype=data.A.dtype,
            chunks=chunks
        )
        start_time = start_time.compute()
    else:
        A_pfb = pfb(data.A, nchan=nchan, ntap=ntap, window=window, fs=data.bandwidth)
        B_pfb = pfb(data.B, nchan=nchan, ntap=ntap, window=window, fs=data.bandwidth)
    return ChannelizedData(
        A_pfb, B_pfb, start_time=start_time,
        feed_poln=data.feed_poln,
        chan_bw=data.bandwidth/nchan, freqs=freqs,
    )


class ChannelizedModel:
    """
    A model representing data which has been channelized using a polyphase filterbank.
    """
    def __init__(self, baseband_model, nchan, ntap=24, window='hamming'):
        """
        Create a channelized model.

        Parameters
        ----------
        baseband_model: The underlying baseband data model
        nchan: Number of channels into which to split the data
        ntap: Number of polyphase filterbank taps
        window: Window function used for polyphase filterbank
                (string interpreted by `scipy.signal.get_window()`)
        """
        self.baseband_model = baseband_model
        self.nchan = nchan
        self.ntap = ntap
        self.window = window

    @property
    def chan_bw(self):
        return self.baseband_model.bandwidth/self.nchan

    def sample(self, n_samples, t_start=None, interp=lerp, dtype=np.float32,
               rng=None, chunks=(-1, 'auto')):
        """
        Simulate a specified number of samples in each channel.

        Parameters
        ----------

        n_samples: The number of samples to use.
        t_start: Time of the first sample, as a Time object. If `None`,
                 the predictor epoch will be used.
        interp: Interpolation function to use. Passed to `BasebandModel.sample()`.
        dtype: Numpy dtype to use for samples.
        """
        nlag = self.nchan*(self.ntap - 1)
        n_baseband = self.nchan*n_samples + nlag

        delayed = isinstance(rng, DelayedRNG)
        if delayed:
            shape = (self.nchan, n_samples)
            chunks = da.core.normalize_chunks(chunks, shape=shape, dtype=dtype)
            chunk_sizes = list(self.nchan*chunk for chunk in chunks[1])
            chunk_sizes[0] += nlag
            chunks_baseband = (tuple(chunk_sizes),)

        if t_start is None:
            t_start = self.baseband_model.predictor.epoch

        offset = t_start.offset - nlag/self.baseband_model.bandwidth
        t_start = Time(t_start.mjd, t_start.second, offset)
        kwargs = {}
        if delayed:
            kwargs['chunks'] = chunks_baseband
        data = self.baseband_model.sample(
            n_baseband, t_start, interp, dtype, rng=rng, **kwargs
        )
        return channelize(data, self.nchan, self.ntap, self.window)

class ChannelizedData:
    """
    Channelized data produced by a polyphase filterbank.
    """
    def __init__(self, A, B, start_time, feed_poln, chan_bw, freqs):
        self.A = A
        self.B = B
        self.start_time = start_time
        self.feed_poln = feed_poln
        self.chan_bw = chan_bw
        self.freqs = freqs

    @property
    def nchan(self):
        return self.A.shape[0]

    @property
    def n_samples(self):
        return self.A.shape[-1]

    @property
    def chunks(self):
        if self.delayed:
            return self.A.chunks
        else:
            return tuple((n,) for n in self.A.shape)

    @property
    def bandwidth(self):
        return self.nchan*self.chan_bw

    @property
    def tspan(self):
        n_samples = self.A.shape[-1]
        sample_freq = np.abs(self.chan_bw)
        return n_samples/sample_freq

    @property
    def t(self):
        sample_freq = np.abs(self.chan_bw)
        return get_time_axis(self.start_time, self.n_samples, sample_freq, self.delayed)

    @property
    def delayed(self):
        return isinstance(self.A, da.Array)

    def compute(self, **kwargs):
        if self.delayed:
            A = self.A.compute(**kwargs)
            B = self.B.compute(**kwargs)
            return ChannelizedData(
                A, B, self.start_time, self.feed_poln, self.chan_bw, self.freqs
            )
        else:
            return self

    def rechunk(self, chunks='auto', **kwargs):
        if self.delayed:
            A = self.A.rechunk(chunks=chunks, **kwargs)
            B = self.B.rechunk(chunks=chunks, **kwargs)
            return ChannelizedData(
                A, B, self.start_time, self.feed_poln, self.chan_bw, self.freqs,
            )
        else:
            return self

    def extract_channel(self, ichan):
        return BasebandData(
            self.A[ichan], self.B[ichan], start_time=self.start_time,
            feed_poln=self.feed_poln, bandwidth=self.chan_bw, obsfreq=self.freqs[ichan],
        )

    def __getitem__(self, ichan):
        return self.extract_channel(ichan)

    def cycfold(self, ncyc, nbin, predictor, use_cuda=have_cuda,
                   n_threads=nb.config.NUMBA_NUM_THREADS):
        """
        Compute the periodic spectrum from channelized data.

        Parameters
        ----------
        ncyc: Number of cyclic channels per filterbank channel
        nbin: Number of phase bins in which to accumulate
        predictor: `PhasePredictor` object to use in computing phases.
        use_cuda: Whether to use CUDA acceleration. Defaults to True if a CUDA
                  device is detected by CuPy. Otherwise defaults to False.
        n_threads: Number of CPU threads to use. Defaults to the total number of
                   CPUs, as detected by Numba. Has no effect if use_gpu is True.
        """
        if use_cuda and have_cuda:
            cycfold = cycfold_gpu
            cycfold_kwargs = {}
        elif use_cuda:
            err = ValueError("use_cuda was specified, but no CUDA device was found")
            raise err
        else:
            cycfold = cycfold_cpu
            cycfold_kwargs = {'n_threads': n_threads}

        freq = np.zeros(ncyc*self.nchan)
        I = np.zeros((ncyc*self.nchan, nbin))
        Q = np.zeros((ncyc*self.nchan, nbin))
        U = np.zeros((ncyc*self.nchan, nbin))
        V = np.zeros((ncyc*self.nchan, nbin))
        chan_data = self.extract_channel(0)
        pspec = cycfold(chan_data, ncyc, nbin, predictor, **cycfold_kwargs)
        if self.nchan % 2 == 0:
            bot = slice(0, ncyc - ncyc//2)
            top = slice(-ncyc//2, None)
            freq[top] = pspec.freq[:ncyc//2] + self.bandwidth
            freq[bot] = pspec.freq[ncyc//2:]
            I[top] = pspec.I[:ncyc//2]
            I[bot] = pspec.I[ncyc//2:]
            Q[top] = pspec.Q[:ncyc//2]
            Q[bot] = pspec.Q[ncyc//2:]
            U[top] = pspec.U[:ncyc//2]
            U[bot] = pspec.U[ncyc//2:]
            V[top] = pspec.V[:ncyc//2]
            V[bot] = pspec.V[ncyc//2:]
            start = 1
            offs = -ncyc//2
            logger.info("Even case")
        else:
            start = 0
            offs = 0
            logger.info("Odd case")
        for ichan in range(start, self.nchan):
            chan_data = self.extract_channel(ichan)
            pspec = cycfold(chan_data, ncyc, nbin, predictor, **cycfold_kwargs)
            sl = slice(ichan*ncyc + offs, (ichan + 1)*ncyc + offs)
            freq[sl] = pspec.freq
            I[sl] = pspec.I
            Q[sl] = pspec.Q
            U[sl] = pspec.U
            V[sl] = pspec.V
        return PeriodicSpectrum(freq, self.start_time, I, Q, U, V)

    def fold(self, nbin, predictor):
        phi = predictor.phase(self.t)
        if self.delayed:
            phi = phi.rechunk(chunks=self.A.chunks[1])
            AA, BB, CR, CI = [], [], [], []
            for phi_blk, A_blk, B_blk in zip(phi.blocks, self.A.blocks, self.B.blocks):
                AA_blk, BB_blk, CR_blk, CI_blk = dask.delayed(fold_channelized, nout=4)(
                    phi_blk, A_blk, B_blk, nbin
                )
                AA.append(da.from_delayed(AA_blk, (nbin,), dtype=self.A.real.dtype))
                BB.append(da.from_delayed(BB_blk, (nbin,), dtype=self.A.real.dtype))
                CR.append(da.from_delayed(CR_blk, (nbin,), dtype=self.A.real.dtype))
                CI.append(da.from_delayed(CI_blk, (nbin,), dtype=self.A.real.dtype))
            AA = da.mean(da.stack(AA), axis=0)
            BB = da.mean(da.stack(BB), axis=0)
            CR = da.mean(da.stack(CR), axis=0)
            CI = da.mean(da.stack(CI), axis=0)
            AA, BB, CR, CI = dask.compute(AA, BB, CR, CI)
        else:
            logger.debug(
                "phi.shape: {}, A.shape: {}, B.shape: {}, nbin: {}",
                phi.shape,
                self.A.shape,
                self.B.shape,
                nbin
            )
            phase = phi % 1
            phase_bin = np.int64(np.round(phase*nbin)) % nbin
            logger.debug(
                "max phase bin: {}, min phase_bin: {}",
                np.min(phase_bin),
                np.max(phase_bin),
            )
            AA, BB, CR, CI = fold_channelized(phi, self.A, self.B, nbin)
        I, Q, U, V = coherence_to_stokes(
            AA, BB, CR, CI, self.feed_poln
        )
        return PeriodicSpectrum(self.freqs, self.start_time, I, Q, U, V)
