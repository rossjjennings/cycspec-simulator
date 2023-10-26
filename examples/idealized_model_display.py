import numpy as np
import matplotlib.pyplot as plt
from cycspec_simulator import (
	TemplateProfile,
	BasebandModel,
	FreqOnlyPredictor,
	ExponentialScatteringModel,
	Time,
	cycfold_cpu,
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
pspec = cycfold_cpu(data, nchan, nbin, predictor)

pc = pspec.plot(shift=0.5, cmap='RdBu_r', sym_lim=True)
plt.colorbar(pc)
plt.show()
