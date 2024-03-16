# See also "Notes on the GUPPI Raw Data Format", S. Ellingson,
# https://www.cv.nrao.edu/~pdemores/GUPPI_Raw_Data_Format/
import numpy as np
import dask
import dask.array as da
import os
import numbers
import warnings

from .metadata import ObservingMetadata
from .baseband import BasebandData
from .time import Time

class GuppiRawHeader:
    def __init__(self, cards):
        self.cards = cards

    def __getitem__(self, key):
        return self.cards[key.upper()]

    def __setitem__(self, key, value):
        if len(key.encode('ascii')) > 8:
            raise ValueError("Keys can have at most 8 characters")
        key = key.upper()
        if isinstance(value, bytes):
            self.cards[key] = value.decode('ascii')
        elif isinstance(value, numbers.Number): # includes booleans
            self.cards[key] = value
        else:
            self.cards[key] = str(value)

    def __str__(self):
        return b"\n".join(self.cards_as_bytes()).decode('ascii')

    def as_bytes(self):
        return b"".join(self.cards_as_bytes())

    def cards_as_bytes(self):
        for key, value in self.cards.items():
            if isinstance(value, bool):
                card = f"{key:<8}= {'T' if value else 'F':>20}"
            elif isinstance(value, numbers.Number):
                card = f"{key:<8}= {value:>20}"
            else:
                card = f"{key:<8}= '{value:<8}'"
            yield f"{card:<80}".encode('ascii')

    @classmethod
    def from_fh(cls, fh):
        cards = {}
        while True:
            try:
                card_bytes = fh.read(80)
                card = card_bytes.decode('ascii')
            except UnicodeDecodeError:
                print(f"current position: {fh.tell()}")
                print(f"card bytes: {card_bytes}")
                raise
            if len(card) < 80:
                raise EOFError("Reached end of file")
            if card.startswith("END"):
                break
            key, value = card.split("=")
            key = key.strip()
            value = value.strip()
            if value in ['T', 'F']:
                value = (value == 'T')
            elif value.startswith("'"):
                value = value.strip("' ")
            else:
                try:
                    value = int(value)
                except ValueError:
                    value = float(value)
            cards[key] = value
        return cls(cards)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as fh:
            header = cls.from_fh(fh)
        return header

def read_headers(filename):
    with open(filename, 'rb') as fh:
        while True:
            try:
                header = GuppiRawHeader.from_fh(fh)
            except EOFError:
                return
            yield header
            fh.seek(header['BLOCSIZE'], os.SEEK_CUR)

class GuppiRaw:
    def __init__(self, headers, data):
        self.headers = headers
        self.data = data

    @property
    def header(self):
        return self.headers[0]

    @property
    def n_blocks(self):
        return len(self.headers)

    @property
    def nchan(self):
        return int(self.header['OBSNCHAN'])

    @property
    def overlap(self):
        return int(self.header['OVERLAP'])

    @property
    def packets_per_block(self):
        return self.headers[1]['PKTIDX'] - self.headers[0]['PKTIDX']

    @property
    def bytes_per_packet(self):
        nbytes = self.bytes_per_block(include_overlap=False)
        return nbytes//self.packets_per_block

    @property
    def bytes_per_sample(self):
        nbits = int(self.header['NBITS'])
        dtype = np.dtype(f"int{nbits}")
        # (2 real / complex) * (2 pols) * (# of channels) * itemsize
        return 2*2*self.nchan*dtype.itemsize

    def bytes_per_block(self, include_overlap=True):
        nbytes_overlap = self.overlap*self.bytes_per_sample
        nbytes = self.header['BLOCSIZE']
        return nbytes if include_overlap else nbytes - nbytes_overlap

    def samples_per_block(self, include_overlap=False):
        # block size in bytes = (# of samples) * (# of bytes/sample)
        nsamp_block = self.bytes_per_block()//self.bytes_per_sample
        return nsamp_block if include_overlap else nsamp_block - self.overlap

def read_raw(filename, use_dask=True, include_overlap=True):
    """
    Read data from a GUPPI Raw file into a GuppiRaw object.

    Parameters
    ----------
    filename: Name of file to read from
    use_dask: If `True`, create a memory-mapped array using Dask.
              If `False`, create an in-memory Numpy array.
    include_overlap: Whether to include the overlap region at the end of each block
    """
    headers = []
    chunks = []
    which_chunk = 0
    with open(filename, 'rb') as fh:
        while True:
            try:
                header = GuppiRawHeader.from_fh(fh)
                headers.append(header)
            except EOFError:
                break

            npol = int(header['NPOL'])
            if npol != 4:
                raise ValueError(f"NPOL = {npol} not supported")
            nbits = int(header['NBITS'])
            dtype = np.dtype(f"int{nbits}")
            nchan = int(header['OBSNCHAN'])
            blocsize = header['BLOCSIZE']
            nsamples = blocsize//(nchan*2*2*dtype.itemsize)

            dtype = np.dtype(f"int{nbits}")
            shape = (nchan, nsamples, 2, 2) # last two axes are polarization, I/Q

            if use_dask:
                delayed = dask.delayed(np.memmap)(
                    filename, mode='r', shape=shape, dtype=dtype, offset=fh.tell()
                )
                chunk = da.from_delayed(delayed, shape=shape, dtype=dtype, name=False)
                fh.seek(blocsize, os.SEEK_CUR)
            else:
                chunk = np.frombuffer(fh.read(blocsize), dtype=dtype).reshape(shape)
            if not include_overlap:
                chunk = chunk[:,:-int(header['OVERLAP'])]
            chunks.append(chunk)
            which_chunk += 1
    if use_dask:
        data = da.concatenate(chunks, axis=1)
    else:
        data = np.concatenate(chunks, axis=1)
    return GuppiRaw(headers, data)

def read(filename):
    raw = read_raw(filename, use_dask=True, include_overlap=True)
    last_overlap = raw.data[:, -raw.overlap:]
    data = da.overlap.trim_overlap(raw.data, depth={1: (0, raw.overlap)})
    complex_data = data[..., 0] + np.complex64(1j)*data[..., 1]
    A = complex_data[..., 0]
    B = complex_data[..., 1]
    start_time = Time(
        raw.header['STT_IMJD'],
        raw.header['STT_SMJD'],
        raw.header['STT_OFFS'],
    )
    feed_poln = raw.header['FD_POLN']
    chan_bw = float(raw.header['OBSBW'])/int(raw.header['OBSNCHAN'])*1e6
    obsfreq = float(raw.header['OBSFREQ'])*1e6
    return BasebandData(A, B, start_time, feed_poln, chan_bw, obsfreq)

def quantize(data, out_dtype=np.int8, autoscale=True):
    stacked = np.stack([data.A, data.B], axis=-1)
    split = np.stack([stacked.real, stacked.imag], axis=-1)
    if autoscale:
        maxval = np.max(np.abs(split))
        mant, expt = np.frexp(maxval)
        nbits = 8*np.dtype(out_dtype).itemsize
        split *= 2**(nbits-1-expt)
    split = split.astype(out_dtype)
    split = split.rechunk((-1, split.chunks[1], -1, -1))
    return split

def write(filename, data, metadata=None, samples_per_block=None, overlap=12288,
          pktsize=8192, out_dtype=np.int8, autoscale=True, **kwargs):
    """
    Write channelized data to a GUPPI raw file.

    Parameters
    ----------
    filename: Name of file to write data to
    data: `ChannelizedData` object
    samples_per_block: Number of samples write in each block. If `None`, will
             be determined automatically from the block structure of the data.
    pktsize: Number of bytes in a "packet". Usually not necessary to change.
    overlap: Number of overlap samples.
    out_dtype: Data type of output. Default is int8 (8 bits).
    autoscale: Whether to automatically scale the data to fit in the range of
              the output data type.
    Additional keyword arguments are stored as header cards in the output file.
    """
    nbytes = np.dtype(out_dtype).itemsize
    nsamples = data.n_samples
    nchan = data.nchan
    bytes_per_sample = nchan*2*2*nbytes
    if not data.delayed:
        A = da.from_array(data.A, name=False)
        B = da.from_array(data.B, name=False)
        data = ChannelizedData(
            A, B, data.start_time, data.feed_poln, data.chan_bw, data.freqs
        )
    if samples_per_block is None and len(data.chunks) == 1:
        # write at least 2 blocks, otherwise DSPSR will have a hard time
        samples_per_block = int(np.ceil(nsamples/2))
    if samples_per_block is not None:
        data = data.rechunk((-1, samples_per_block))
    if metadata is None:
        metadata = ObservingMetadata.default()
    offset = data.t[0].offset.compute()

    header = GuppiRawHeader({
        'SRC_NAME': metadata.src_name,
        'TELESCOP': metadata.telescope,
        'FRONTEND': metadata.frontend,
        'BACKEND': metadata.backend,
        'RA_STR': metadata.ra_str,
        'DEC_STR': metadata.dec_str,
        'OBSERVER': metadata.observer,
        'OBSFREQ': f'{data.freqs[data.freqs.size//2]/1e6:.16g}',
        'OBSBW': f'{data.nchan*data.chan_bw/1e6:.16g}',
        'TBIN': f'{1/data.chan_bw:.16g}',
        'STT_IMJD': data.t.mjd,
        'STT_SMJD': data.t.second + int(offset),
        'STT_OFFS': offset - int(offset),
        'PKTIDX': 0, # to be filled in later
        'PKTSIZE': pktsize,
        'PKTFMT': '1SFA',
        'NRCVR': '2',
        'NPOL': '4',
        'POL_TYPE': 'AABBCRCI',
        'FD_POLN': data.feed_poln,
        'NBITS': 8*nbytes,
        'OBSNCHAN': f'{nchan}',
        'BLOCSIZE': 0, # to be filled in later
        'OVERLAP': overlap,
    })

    for key, value in kwargs.items():
        header[key.upper()[:8]] = value

    quantized_data = quantize(data, out_dtype, autoscale)
    quantized_data = da.overlap.overlap(
        quantized_data,
        depth={1: (0, overlap)},
        boundary=None
    )

    with open(filename, 'wb') as fh:
        pktidx = 0
        for iblock, block in enumerate(quantized_data.blocks.ravel()):
            print(f"Writing block {iblock} with block size {block.size}")
            header['PKTIDX'] = pktidx
            header['BLOCSIZE'] = block.size*nbytes
            pktidx += block.size*nbytes//pktsize
            for card in header.cards_as_bytes():
                fh.write(card)
            fh.write(b"END" + b" "*77)
            fh.write(block.compute().tobytes())
