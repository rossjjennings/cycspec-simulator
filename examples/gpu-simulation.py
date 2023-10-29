#/usr/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from cycspec_simulator import (
    TemplateProfile,
    BasebandModel,
    FreqOnlyPredictor,
    PolynomialPredictor,
    ExponentialScatteringModel,
    ObservingMetadata,
    Time,
    cycfold_gpu,
    cycfold_gpu_sharedmem,
    guppi_raw,
)

def simulate(template_file, nchan, chan_bw, obsfreq, scattering_time, n_samples, ncyc, nbin, sharedmem=False):
    print("Loading template")
    template = TemplateProfile.from_file(template_file)
    template.normalize()
    template.make_posdef()

    metadata = ObservingMetadata.from_file(template_file)
    metadata.observer = "cycspec-simulator"

    print("Loading polyco")
    predictor = PolynomialPredictor.from_file("polyco-B1937+21-60000.dat")
    model = BasebandModel(template, nchan=nchan, chan_bw=chan_bw, predictor=predictor, obsfreq=obsfreq)

    print("Constructing scattering model")
    scattering_model = ExponentialScatteringModel(
        scattering_time=scattering_time, nchan=nchan, chan_bw=model.chan_bw, obsfreq=obsfreq, cutoff=20
    )
    pattern = scattering_model.realize()

    print("Creating simulated data")
    t_start = Time(60000, 1800, -(pattern.impulse_response.shape[-1]-1)/chan_bw)
    data = model.sample(n_samples + pattern.impulse_response.shape[-1] - 1, t_start=t_start)
    data = pattern.scatter(data)
    print(f"dtype of data: {data.A.dtype}")

    print("Folding cyclic spectrum")
    if sharedmem:
        return cycfold_gpu_sharedmem(data, ncyc, nbin, predictor)
    else:
        return cycfold_gpu(data, ncyc, nbin, predictor)

def plot(pspec):
    pc = pspec.plot(shift=0.25, cmap='RdBu_r', sym_lim=True)
    plt.colorbar(pc)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-file', type=str,
                        default="B1937+21.Rcvr1_2.GUPPI.15y.x.sum.sm",
                        help="PSRFITS file with template profile")
    parser.add_argument('-c', '--nchan', type=int, default=1,
                        help="Number of input channels (PFB channels) to simulate")
    parser.add_argument('-w', '--chan-bw', type=float, default=1.5625,
                        help="Bandwidth of each simulated input channel (MHz)")
    parser.add_argument('-f', '--obsfreq', type=float, default=1500.,
                        help="Treat simulated data as having this center frequency (MHz)")
    parser.add_argument('-s', '--scattering-time', type=float, default=40.,
                        help="Scattering time in microseconds")
    parser.add_argument('-n', '--n-samples', type=str, default="4M",
                        help="Number of samples (1k=1024, 1M=1024k)")
    parser.add_argument('-k', '--ncyc', type=int, default=1024,
                        help="Number of output channels per input channel")
    parser.add_argument('-b', '--nbin', type=int, default=1024,
                        help="Number of phase bins in output periodic spectrum")
    parser.add_argument('-y', '--no-plot', action='store_true',
                        help="Don't plot output periodic spectrum")
    parser.add_argument('--sharedmem', action='store_true',
                        help="Use shared memory implementation")
    args = parser.parse_args()

    if args.n_samples.endswith("k"):
        args.n_samples = int(args.n_samples.strip("k"))*2**10
    elif args.n_samples.endswith("M"):
        args.n_samples = int(args.n_samples.strip("M"))*2**20
    else:
        args.n_samples = int(args.n_samples)
    print(f"Number of samples: {args.n_samples}")
    args.chan_bw *= 1e6 # MHz -> Hz
    args.obsfreq *= 1e6 # MHz -> Hz
    args.scattering_time /= 1e6 # μs -> s
    pspec = simulate(args.template_file, args.nchan, args.chan_bw, args.obsfreq,
                     args.scattering_time, args.n_samples, args.ncyc, args.nbin, args.sharedmem)
    if not args.no_plot:
        plot(pspec)
