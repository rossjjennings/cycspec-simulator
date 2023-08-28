# See also "Notes on the GUPPI Raw Data Format", S. Ellingson,
# https://www.cv.nrao.edu/~pdemores/GUPPI_Raw_Data_Format/
import numpy as np
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
            card = fh.read(80).decode('ascii')
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
            else:
                yield header
                fh.seek(header['BLOCSIZE'], os.SEEK_CUR)

def read_frame(fh):
    header = GuppiRawHeader.from_fh(fh)
    data_start = fh.tell()

    npol = int(header['NPOL'])
    if npol != 4:
        raise ValueError(f"NPOL = {npol} not supported")

    nbits = int(header['NBITS'])
    dtype = np.dtype(f"int{nbits}")

    nchan = int(header['OBSNCHAN'])
    overlap = int(header['OVERLAP'])
    blocsize = header['BLOCSIZE']
    frame = np.frombuffer(fh.read(blocsize), dtype=dtype)
    frame = frame.reshape(nchan, -1, 2, 2)
    return header, frame

def read_dask():
    pass

def write_frame(arr, header, fh):
    for card in header.cards_as_bytes():
        fh.write(card)
