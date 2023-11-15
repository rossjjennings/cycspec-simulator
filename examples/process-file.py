#!/usr/bin/env python
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
    cycfold_cpu,
    guppi_raw,
)

def process(infile, polyco, ncyc, nbin):
    print(f"Loading polynomial coefficients from {polyco}")
    predictor = PolynomialPredictor.from_file(polyco)

    print(f"Reading data from {infile}")
    data = guppi_raw.read(infile, use_dask=False)

    print("Folding cyclic spectrum")
    return cycfold_cpu(data, ncyc, nbin, predictor)

def plot(pspec):
    pc = pspec.plot(shift=0.25, cmap='RdBu_r', sym_lim=True)
    plt.colorbar(pc)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--polyco', type=str, default=1024,
                        help="Polyco file to use for folding")
    parser.add_argument('-k', '--ncyc', type=int, default=1024,
                        help="Number of output channels per input channel")
    parser.add_argument('-b', '--nbin', type=int, default=1024,
                        help="Number of phase bins in output periodic spectrum")
    parser.add_argument('-y', '--no-plot', action='store_true',
                        help="Don't plot output periodic spectrum")
    parser.add_argument('infile', type=str, help="GUPPI raw file to fold")
    args = parser.parse_args()
    pspec = process(args.infile, args.polyco, args.ncyc, args.nbin)
    if not args.no_plot:
        plot(pspec)
