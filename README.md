# Cycspec Simulator
Cycspec Simulator is a set of tools for simulating and testing various approaches to cyclic spectral analysis for pulsar timing. It is structured as an installable Python package. To use it, first clone this repository:
```bash
git clone https://github.com/rossjjennings/cycspec-simulator.git
```
You can then install the package as follows:
```bash
cd cycspec-simulator
pip install .
```
Make sure you are installing into a virtual environment with Python 3.9 or newer. Pip should automatically resolve and install the dependencies, which are listed in `pyproject.toml`.

## Example notebooks
The Jupyter notebooks in the `examples/` directory illustrate some things you can do with this package. In particular:
- `benchmarking.ipynb` compares the performance of different cyclic spectroscopy implementations, using data without scattering added.
- `simulate.ipynb` generates simulated data with scattering added, folds it to produce a periodic spectrum, and writes the raw data to a file in the [GUPPI raw format](https://www.cv.nrao.edu/~pdemores/GUPPI_Raw_Data_Format/).

## Generating simulated data from the command line
You can create simulated data from the command line using the `cycspec-make-raw` executable (equivalent to `python make_raw.py`), which is installed by `pip` when you install the package.

This command takes as input a template profile in PSRFITS format, and a [TEMPO](https://tempo.sourceforge.net/)-format [`polyco.dat` file](https://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt) describing the pulse phase as a function of time. Appropriate template profiles can be found under `narrowband/templates` in the [NANOGrav 15-year dataset](https://doi.org/10.5281/zenodo.7967584), and a single example is included here as `examples/B1937+21.Rcvr1_2.GUPPI.15y.x.sum.sm`. The example polyco file `examples/polyco-B1937+21-60000.dat` was generated from `examples/B1937+21_pred.par` using TEMPO, with the following command:
```bash
tempo -f B1937+21_pred.par -ZPSR=B1937+21 -ZOBS=GB -ZSTART=60000 -ZTOBS=1H -ZFREQ=1500.78125 -ZOUT=polyco-B1937+21-60000.dat
```

An example command that uses `cycspec-make-raw` to create a raw data file is the following:
```bash
cycspec-make-raw -t B1937+21.Rcvr1_2.GUPPI.15y.x.sum.sm -p polyco-B1937+21-60000.dat -o B1937+21-test.raw -n 4194304 -b 1.5625 -f 1500.78125 -s 4e-5
```
The above command uses the included example template and polyco files to create 2^22 = 4 194 304 samples, at a bandwidth of 1.5625 MHz (corresponding to a total integration time of approximately 2.68 s) and a nominal center frequency of 1500.78125 MHz, including scattering with a scattering time of 40 μs (i.e., 4 × 10^-5 seconds).

To get the full list of command-line arguments, run `cycspec-make-raw -h`.

## Using from Python
For more control over the simulation, you can write your own Python code using the provided classes, either in a script or in a notebook. For this purpose, the most important class is `BasebandModel`. Once you have constructed a `BasebandModel` instance, you can use its `sample()` method to create simulated data.

Creating a `BasebandModel` requires a template profile (class `TemplateProfile`), and a phase predictor (such as a `PolynomialPredictor` or `FreqOnlyPredictor`) with a `phase()` method that can be used to compute the pulse phase at any particular time. A `TemplateProfile` can be constructed either directly from Numpy arrays giving the Stokes parameters I, Q, U, and V (or total intensity, I, only) as a function of phase, or from a template file. Similarly, a phase predictor can be a `PolynomialPredictor` derived from a polyco file, or a `FreqOnlyPredictor` depending only on the pulse frequency and phase zero point.

Once the data is generated, scattering can be applied using the `scatter()` method of a `ScintillationPattern` instance, and the data can be folded to form a periodic spectrum using the `cycspec.pspec_numba()` function, or written to disk using the `guppi_raw.write()` function. Putting it all together, a script to create simulated data based on a simple Gaussian pulse, and fold it to produce a periodic spectrum estimate, might look like the following:
```python
import numpy as np
import matplotlib.pyplot as plt
from cycspec_simulator import (
	TemplateProfile,
	BasebandModel,
	FreqOnlyPredictor,
	ExponentialScatteringModel,
	Time,
	pspec_numba,
)

phase = np.linspace(0, 1, 2048, endpoint=False)
width = 0.01
I = np.exp(-phase**2/(2*width**2))
template = TemplateProfile(I)

pulse_freq = 500 # Hz
epoch = Time(60000, 0, 0) # MJD, UTC second, fraction
predictor = FreqOnlyPredictor(pulse_freq, epoch)

chan_bw = 1.0e6 # Hz
obsfreq = 1.5e9 # Hz

model = BasebandModel(template, predictor, chan_bw, obsfreq=obsfreq)
data = model.sample(2**20)

scattering_model = ExponentialScatteringModel(
	scattering_time=2e-5, chan_bw=model.chan_bw, obsfreq=obsfreq
)
pattern = scattering_model.realize()
data = pattern.scatter(data)

nchan = 512
nbin = 1024
pspec = pspec_numba(data, nchan, nbin, predictor)

pc = pspec.plot(shift=0.5, cmap='RdBu_r', sym_lim=True)
plt.colorbar(pc)
plt.show()
```

A somewhat more realistic simulation based on a real pulse profile and polyco file, which writes the resulting data to a GUPPI raw file, is not much more complicated:
```python
import numpy as np
from cycspec_simulator import (
	TemplateProfile,
	BasebandModel,
	PolynomialPredictor,
	ExponentialScatteringModel,
	ObservingMetadata,
	guppi_raw,
)

template_file = "B1937+21.Rcvr1_2.GUPPI.15y.x.sum.sm"
template = TemplateProfile.from_file(template_file)
template.normalize()
template.make_posdef()

polyco_file = "polyco-B1937+21-60000.dat"
predictor = PolynomialPredictor.from_file(polyco_file)

chan_bw = 1.5625e6 # Hz
nchan = 2
obsfreq = 1.5e9 # Hz
model = BasebandModel(template, predictor, chan_bw, nchan=nchan, obsfreq=obsfreq)
data = model.sample(2**22)

scattering_model = ExponentialScatteringModel(
	scattering_time=2e-5, # s
	chan_bw=model.chan_bw,
	nchan=nchan,
	obsfreq=obsfreq,
)
pattern = scattering_model.realize()
data = pattern.scatter(data)

metadata = ObservingMetadata.from_file(template_file)
metadata.observer = "cycspec-simulator"
guppi_raw.write("simulated-data.raw", data, metadata=metadata)
```
