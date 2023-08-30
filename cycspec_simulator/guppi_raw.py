# See also "Notes on the GUPPI Raw Data Format", S. Ellingson,
# https://www.cv.nrao.edu/~pdemores/GUPPI_Raw_Data_Format/
import numpy as np
import dask
import dask.array as da
import os
import numbers

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

def read(filename, use_dask=True):
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
            if which_chunk > 0:
                chunk = chunk[:,int(header['OVERLAP']):]
            chunks.append(chunk)
            which_chunk += 1
    if use_dask:
        data = da.concatenate(chunks, axis=1)
    else:
        data = np.concatenate(chunks, axis=1)
    return headers, data

def quantize(data, out_dtype=np.int8, autoscale=True):
    stacked = np.stack([data.A, data.B], axis=-1)
    split = np.stack([stacked.real, stacked.imag], axis=-1)
    if autoscale:
        maxval = np.max(np.abs(split))
        mant, expt = np.frexp(maxval)
        nbits = 8*np.dtype(out_dtype).itemsize
        split *= 2**(nbits-1-expt)
    return (split).astype(out_dtype)

def write(filename, data, header, blocsize=None, overlap=0, out_dtype=np.int8, autoscale=True):
    """
    Write data to a GUPPI raw file.
    Currently very limited -- can only write a single frame with one channel.
    """
    if not isinstance(header, GuppiRawHeader):
        # assume it's a dict, or dict-like
        header = GuppiRawHeader(header)
    nchan = 1
    nbytes = np.dtype(out_dtype).itemsize
    if blocsize is None:
        datasize = data.t.shape[0]*nchan*2*2*nbytes
        blocsize = datasize

    header['NPOL'] = '4'
    header['NBITS'] = 8*nbytes
    header['OBSNCHAN'] = str(nchan)
    header['BLOCSIZE'] = blocsize
    header['OVERLAP'] = overlap

    quantized_data = quantize(data, out_dtype, autoscale)

    with open(filename, 'wb') as fh:
        for card in header.cards_as_bytes():
            fh.write(card)
        fh.write(b"END" + b" "*77)
        fh.write(quantized_data.tobytes())
