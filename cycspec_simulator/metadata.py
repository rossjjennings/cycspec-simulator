from astropy.io import fits
from astropy.coordinates import SkyCoord

class ObservingMetadata:
    def __init__(self, src_name, telescope, frontend, backend, observer, location):
        self.src_name = src_name
        self.telescope = telescope
        self.frontend = frontend
        self.backend = backend
        self.observer = observer
        self.location = location

    @classmethod
    def default(cls):
        return cls(
            src_name="simulated-pulsar",
            telescope="fake-telescope",
            frontend="fake-receiver",
            backend="BOGUS",
            observer="cycspec-simulator",
            location=SkyCoord(0, 0, unit="degree"),
        )

    @classmethod
    def from_file(cls, filename):
        hdul = fits.open(filename)
        header = hdul['PRIMARY'].header
        src_name = header['SRC_NAME']
        telescope = header['TELESCOP']
        frontend = header['FRONTEND']
        backend = header['BACKEND']
        observer = header['OBSERVER']
        ra = header['RA']
        dec = header['DEC']
        hdul.close()

        location = SkyCoord(ra, dec, unit=("hourangle", "degree"))
        return cls(src_name, telescope, frontend, backend, observer, location)

    @property
    def ra_str(self):
        return self.location.ra.to_string(unit="hourangle", sep=':', precision=4)

    @property
    def dec_str(self):
        return self.location.dec.to_string(unit="degree", sep=':', precision=4)
