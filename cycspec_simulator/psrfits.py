import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Angle

from textwrap import dedent
from datetime import datetime

from .phase_predictor import FreqOnlyPredictor, PolynomialPredictor

# PSRFITS is a defined subset of the FITS image format used for storing
# radio pulsar observations, including fold mode and search mode data.
# Documentation of the header cards and table columns used in the PSRFITS
# format can be found at the ATNF PSRFITS definition page:
# https://www.atnf.csiro.au/research/pulsar/psrfits_definition/PsrfitsDocumentation.html

# The "blank" PSRFITS file distributed with the PSRCHIVE source code
# (Base/Formats/PSRFITS/psrheader.fits) is also useful for determining
# appropriate default values.

def to_hdulist(pspec):
    """
    Convert an periodic spectrum to a PSRFITS HDU list for saving.

    Parameters
    ----------
    pspec: PeriodicSpectrum
        Periodic spectrum to represent in PSRFITS format.

    Returns
    -------
    hdul: astropy.fits.HDUList
        FITS HDU list representing the periodic spectrum in PSRFITS format.
        The data can be written to file using the `hdul.writeto()` method.
    """
    hdus = []
    hdus.append(construct_primary_hdu(pspec))
    hdus.append(construct_history_hdu(pspec))
    if isinstance(pspec.predictor, PolynomialPredictor):
        hdus.append(construct_polyco_hdu(pspec.predictor))
        period = None
    elif isinstance(pspec.predictor, FreqOnlyPredictor):
        period = 1/pspec.predictor.f0
    hdus.append(construct_subint_hdu(pspec, period))
    return fits.HDUList(hdus)

def construct_primary_hdu(pspec):
    """
    Construct the primary HDU for a FITS file representing this periodic spectrum.
    The primary HDU of a PSRFITS file contains only header information, with the
    data portion being empty, and most of the data instead being stored in the
    SUBINT HDU. This is a departure from the most common practice for optical
    FITS images, where the main image data is stored in the data portion of
    the primary HDU.

    Parameters
    ----------
    pspec: PeriodicSpectrum
        Periodic spectrum to represent in PSRFITS format.

    Returns
    -------
    primary_hdu: astropy.fits.PrimaryHDU
        Primary FITS header and data unit (HDU) needed to represent the
        periodic spectrum in PSRFITS format. Must be combined with required
        extension HDUs (at least HISTORY and SUBINT, and usually either
        POLYCO or T2PREDICT) to make a well-formed PSRFITS file.
    """
    primary_hdu = fits.PrimaryHDU()
    fits_description = dedent("""
    FITS (Flexible Image Transport System) format is defined in 'Astronomy
    and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H.
    Contact the NASA Science Office of Standards and Technology for the
    FITS Definition document #100 and other FITS information.
    """)[1:]
    for line in fits_description.split('\n'):
        primary_hdu.header['comment'] = line

    nchan, = pspec.freq.shape
    header_cards = {
        # Setting HDRVER = '6.1' makes PSRCHIVE look for (and not find)
        # a column named 'REF_FREQ' in the generated file.
        # So, leaving as '5.4' for now.
        'HDRVER': "5.4",
        'FITSTYPE': "PSRFITS",
        'DATE': datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S'),
        'OBSERVER': pspec.metadata.observer,
        'PROJID': "",
        'TELESCOP': pspec.metadata.telescope,
        'ANT_X': "*",
        'ANT_Y': "*",
        'ANT_Z': "*",
        'FRONTEND': pspec.metadata.frontend,
        'IBEAM': "",
        'NRCVR': 2 if pspec.full_stokes else 1,
        'FD_POLN': "LIN",
        'FD_HAND': "*",
        'FD_SANG': "*",
        'FD_XYPH': "*",
        'BACKEND': pspec.metadata.backend,
        'BECONFIG': "",
        'BE_PHASE': "*",
        'BE_DCC': "*",
        'BE_DELAY': "*",
        'TCYCLE': "*",
        'OBS_MODE': "PSR",
        'DATE-OBS': datetime.strftime(
            Time(pspec.start_time.to_mjd(), format='mjd').to_datetime(),
            '%Y-%m-%dT%H:%M:%S'),
        'OBSFREQ': np.mean(pspec.freq[1:])/1e6,
        'OBSBW': nchan*(pspec.freq[1] - pspec.freq[0])/1e6,
        'OBSNCHAN': nchan,
        'CHAN_DM': "*",
        'PNT_ID': "",
        'SRC_NAME': pspec.metadata.src_name,
        'COORD_MD': "J2000",
        'EQUINOX': 2000.0,
        'RA': pspec.metadata.location.ra.deg,
        'DEC': pspec.metadata.location.dec.deg,
        'BMAJ': "*",
        'BMIN': "*",
        'BPA': "*",
        'STT_CRD1': "",
        'STT_CRD2': "",
        'TRK_MODE': "",
        'STP_CRD1': "",
        'STP_CRD2': "",
        'SCANLEN': "*",
        'FD_MODE': "",
        'FA_REQ': "*",
        'CAL_MODE': "",
        'CAL_FREQ': "*",
        'CAL_DCYC': "*",
        'CAL_PHS': "*",
        'CAL_NPHS': "*",
        'STT_IMJD': pspec.start_time.mjd,
        'STT_SMJD': pspec.start_time.second,
        'STT_OFFS': pspec.start_time.offset,
        'STT_LST': "*",
    }

    for key, value in header_cards.items():
        primary_hdu.header[key] = value

    return primary_hdu

def construct_history_hdu(pspec):
    """
    Construct the HISTORY HDU for a FITS file representing this periodic spectrum.
    The HISTORY HDU contains a table that subsequent modifications to the file
    should add to rather than replace. Currently, this implementation creates only
    the initial row.

    Parameters
    ----------
    pspec: PeriodicSpectrum
        Periodic spectrum to represent in PSRFITS format.

    Returns
    -------
    history_hdu: astropy.fits.BinTableHDU
        FITS HISTORY header and data unit (HDU) used together with other HDUs
        to represent the periodic spectrum in PSRFITS format. This is an
        extension HDU and must be used together with a primary HDU.
    """
    columns = {
        ('DATE_PRO', 'S24'): datetime.strftime(datetime.now(), '%a %b %d %H:%M:%S %Y'),
        ('PROC_CMD', 'S256'): "UNKNOWN",
        ('SCALE', 'S8',): "FluxDen", # used for uncalibrated data
        ('POL_TYPE', 'S8'): "AABBCRCI" if pspec.full_stokes else "AA+BB",
        ('NSUB', '>i4'): 1,
        ('NPOL', '>i2'): 4 if pspec.full_stokes else 1,
        ('NBIN', '>i2'): pspec.nbin,
        ('NBIN_PRD', '>i2'): pspec.nbin,
        ('TBIN', '>f8'): 1.0,
        ('CTR_FREQ', '>f8'): np.mean(pspec.freq[1:])/1e6,
        ('NCHAN', '>i4'): pspec.freq.shape[0],
        ('CHAN_BW', '>f8'): (pspec.freq[1] - pspec.freq[0])/1e6,
        ('DM', '>f8'): 0.0,
        ('RM', '>f8'): 0.0,
        ('PR_CORR', '>i2'): 0,
        ('FD_CORR', '>i2'): 0,
        ('BE_CORR', '>i2'): 0,
        ('RM_CORR', '>i2'): 0,
        ('DEDISP', '>i2'): 0,
        ('DDS_MTHD', 'S32'): 'UNSET',
        ('SC_MTHD', 'S32'): 'NONE',
        ('CAL_MTHD', 'S32'): 'NONE',
        ('CAL_FILE', 'S256'): 'NONE',
        ('RFI_MTHD', 'S32'): 'NONE',
        ('RM_MODEL', 'S32'): 'NONE',
        ('AUX_RM_C', '>i2'): 0,
        ('DM_MODEL', 'S32'): 'NONE',
        ('AUX_DM_C', '>i2'): 0,
    }

    header_cards = {
        'EXTNAME': "HISTORY",
        'EXTVER': 1,
        'TUNIT9': "s",
        'TUNIT10': "MHz",
        'TUNIT12': "MHz",
        'TUNIT13': "pc cm-3",
        'TUNIT14': "rad",
    }

    table = np.array([tuple(columns.values())], dtype=list(columns.keys()))
    history_hdu = fits.BinTableHDU(data=table)

    for key, value in header_cards.items():
        history_hdu.header[key] = value

    return history_hdu

def construct_subint_hdu(pspec, period=None):
    """
    Construct the SUBINT HDU for a FITS file representing this periodic spectrum.
    The SUBINT HDU is the main data portion of the file, and contains a
    representation of the periodic spectrum in packed 16-bit integer format,
    together with

    Parameters
    ----------
    pspec: PeriodicSpectrum
        Periodic spectrum to represent in PSRFITS format.
    period: float, optional
        Folding period, useful when no POLYCO or T2PREDICT HDU is present.
        If specified, the PERIOD column will be added to the SUBINT HDU.

    Returns
    -------
    subint_hdu: astropy.fits.BinTableHDU
        FITS SUBINT header and data unit (HDU) used together with other HDUs
        to represent the periodic spectrum in PSRFITS format. This is an
        extension HDU and must be used together with a primary HDU.
    """
    if pspec.full_stokes:
        data = np.stack([pspec.I, pspec.Q, pspec.U, pspec.V])
    else:
        data = pspec.I[np.newaxis, ...]
    data = data[np.newaxis, ...]
    mins = np.min(data, axis=-1)
    maxes = np.max(data, axis=-1)
    mant, expt = np.frexp(maxes - mins)
    offsets = (mins + maxes)/2.
    scales = 2.**(expt-16)
    data -= offsets[..., np.newaxis]
    data /= scales[..., np.newaxis]
    data = np.rint(data).astype('i2')

    npol = 4 if pspec.full_stokes else 1
    nchan, = pspec.freq.shape

    columns = {
        ('INDEXVAL', '>f8'): 0.0,
        ('TSUBINT', '>f8'): 1.0,
        ('OFFS_SUB', '>f8'): 0.0,
        ('LST_SUB', '>f8'): 0.0,
        ('RA_SUB', '>f8'): pspec.metadata.location.ra.deg,
        ('DEC_SUB', '>f8'): pspec.metadata.location.dec.deg,
        ('GLON_SUB', '>f8'): 0.0,
        ('GLAT_SUB', '>f8'): 0.0,
        ('FD_ANG', '>f4'): 0.0,
        ('POS_ANG', '>f4'): 0.0,
        ('PAR_ANG', '>f4'): 0.0,
        ('TEL_AZ', '>f4'): 0.0,
        ('TEL_ZEN', '>f4'): 0.0,
        ('AUX_DM', '>f8'): 0.0,
        ('AUX_RM', '>f8'): 0.0,
        ('PERIOD', '>f8'): period,
        ('DAT_FREQ', '>f8', (nchan,)): pspec.freq/1e6,
        ('DAT_WTS', '>f4', (nchan,)): np.ones_like(pspec.freq, dtype='>f4'),
        ('DAT_OFFS', '>f4', (npol*nchan,)): offsets.reshape(1, -1),
        ('DAT_SCL', '>f4', (npol*nchan,)): scales.reshape(1, -1),
        ('DATA', '>i2', (npol, nchan, pspec.nbin)): data,
    }
    if not period:
        del columns[('PERIOD', '>f8')]

    header_cards = {
        'EXTNAME': "SUBINT",
        'EXTVER': 1,
        'EPOCHS': "VALID",
        'INT_TYPE': "TIME",
        'INT_UNIT': "SEC",
        'SCALE': "FluxDen",
        'POL_TYPE': "AABBCRCI" if pspec.full_stokes else "AA+BB",
        'NPOL': 4 if pspec.full_stokes else 1,
        'TBIN': 1.0,
        'NBIN': pspec.nbin,
        'NBIN_PRD': pspec.nbin,
        'PHS_OFFS': "*",
        'NBITS': 1,
        'ZERO_OFF': "*",
        'SIGNINT': 0,
        'NSUBOFFS': "*",
        'NCHAN': nchan,
        'CHAN_BW': (pspec.freq[1] - pspec.freq[0])/1e6,
        'DM': 0.0,
        'RM': 0.0,
        'NCHNOFFS': "*",
        'NSBLK': 1,
        'NSTOT': "*",
        'TUNIT2': "s",
        'TUNIT3': "s",
        'TUNIT4': "s",
        'TUNIT5': "deg",
        'TUNIT6': "deg",
        'TUNIT7': "deg",
        'TUNIT8': "deg",
        'TUNIT9': "deg",
        'TUNIT10': "deg",
        'TUNIT11': "deg",
        'TUNIT12': "deg",
        'TUNIT13': "deg",
        'TUNIT14': "pc cm-3",
        'TUNIT15': "rad m-2",
    }

    if period:
        header_cards['TUNIT16'] = "s"
        idx_offs = 1
    else:
        idx_offs = 0

    header_cards.update({
        f'TUNIT{16+idx_offs}': "MHz",
        f'TUNIT{20+idx_offs}': "Jy",
    })

    table = np.array([tuple(columns.values())], dtype=list(columns.keys()))
    subint_hdu = fits.BinTableHDU(data=table)

    for key, value in header_cards.items():
        subint_hdu.header[key] = value

    return subint_hdu

def construct_polyco_hdu(predictor):
    """
    Construct a POLYCO HDU representing this predictor object.

    Parameters
    ----------
    predictor: PolynomialPredictor
        The TEMPO-style polynomial predictor (polyco) to represent.

    Returns
    -------
    polyco_hdu: astropy.io.fits.BinTableHDU
        FITS POLYCO header and data unit (HDU) representing the predictor.
        This is an extension HDU and must be used together with a primary HDU.
    """
    rows = []
    for segment in predictor.segments:
        columns = {
            ('DATE_PRO', 'S24'): segment.date_produced,
            ('POLYVER', 'S16'): segment.version,
            ('NSPAN', '>i2'): segment.span,
            ('NCOEF', '>i2'): segment.coeffs.shape[0],
            ('NPBLK', '>i2'): 1,
            ('NSITE', 'S8'): segment.site,
            ('REF_FREQ', '>f8'): segment.ref_freq,
            ('PRED_PHS', '>f8'): segment.start_phase,
            ('REF_MJD', '>f8'): segment.epoch.to_mjd(),
            ('REF_PHS', '>f8'): segment.ref_phase,
            ('REF_F0', '>f8'): segment.ref_f0,
            ('LGFITERR', '>f8'): segment.log10_fit_err,
            ('COEFF', '>f8', (15,)): segment.coeffs,
        }
        row = np.array([tuple(columns.values())], dtype=list(columns.keys()))
        rows.append(row[0]) # extract the record from the array
    table = np.stack(rows)

    header_cards = {
        'EXTNAME': "POLYCO",
        'EXTVER': 1,
        'TUNIT7': "MHz",
        'TUNIT11': "Hz",
    }

    polyco_hdu = fits.BinTableHDU(data=table)

    for key, value in header_cards.items():
        polyco_hdu.header[key] = value

    return polyco_hdu
