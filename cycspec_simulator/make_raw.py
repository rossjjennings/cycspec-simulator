import numpy as np

from .template_profile import TemplateProfile
from .baseband import BasebandModel
from .phase_predictor import FreqOnlyPredictor, PolynomialPredictor
from .scattering import ExponentialScatteringModel
from .time import Time
from .metadata import ObservingMetadata
from .guppi_raw import write

def make_raw(template_file, polyco_file, out_file, n_samples, chan_bw,
             nchan, obsfreq, scattering_time, start_mjd, start_second):
    """
    Create a file with simulated data in GUPPI Raw format.

    Parameters
    ----------
    template_file: Name of PSRFITS file containing template profile
    polyco_file: Name of TEMPO polyco file containing phase predictor
    out_file: Name of file to write output to
    n_samples: Number of samples to simulate
    chan_bw: Channel bandwidth of simulated data (Hz)
    obsfreq: Observing frequency of simulated data (Hz)
    scattering_time: Scattering time used for ISM model
    start_mjd: MJD of data start (if `None`, epoch from polyco file will be used)
    start_second: UTC second of data start (if `None`, epoch from polyco file will be used)
    """
    template = TemplateProfile.from_file(template_file)
    template.normalize()
    template.make_posdef()

    predictor = PolynomialPredictor.from_file(polyco_file)
    model = BasebandModel(
        template,
        predictor=predictor,
        chan_bw=chan_bw,
        nchan=nchan,
        obsfreq=obsfreq,
    )
    scattering_model = ExponentialScatteringModel(
        scattering_time=scattering_time,
        chan_bw=model.chan_bw,
        nchan=nchan,
        obsfreq=obsfreq,
        cutoff=20,
    )
    pattern = scattering_model.realize()

    if start_mjd is None:
        start_mjd = predictor.epoch.mjd
    if start_second is None:
        start_second = predictor.epoch.second
        start_second += int(np.round(predictor.epoch.offset))
    n_extra_samples = pattern.impulse_response.shape[-1] - 1
    t_start = Time(start_mjd, start_second, -n_extra_samples/chan_bw)
    data = model.sample(n_samples + n_extra_samples, t_start=t_start)
    data = pattern.scatter(data)

    metadata = ObservingMetadata.from_file(template_file)
    metadata.observer = "cycspec-simulator"
    write(out_file, data, metadata=metadata)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-file', type=str, help="Template file (PSRFITS format)")
    parser.add_argument('-p', '--polyco-file', type=str, help="TEMPO Polyco file")
    parser.add_argument('-o', '--out-file', type=str, help="Output file name")
    parser.add_argument('-n', '--n-samples', type=int, help="Number of samples to simulate")
    parser.add_argument('-c', '--nchan', type=int, default=1, help="Number of channels to simulate")
    parser.add_argument('-b', '--bandwidth', type=float, help="Channel bandwidth (MHz)")
    parser.add_argument('-f', '--obsfreq', type=float, help="Observing frequency (MHz)")
    parser.add_argument('-s', '--scattering-time', type=float, help="Scattering time (s)")
    parser.add_argument('-m', '--start-mjd', type=int, default=None, help="MJD of data start")
    parser.add_argument('-z', '--start-second', type=int, default=None, help="UTC second of data start")
    args = parser.parse_args()

    make_raw(
        args.template_file,
        args.polyco_file,
        args.out_file,
        args.n_samples,
        args.bandwidth*1e6,
        args.nchan,
        args.obsfreq*1e6,
        args.scattering_time,
        args.start_mjd,
        args.start_second
    )

if __name__ == "__main__":
    main()
