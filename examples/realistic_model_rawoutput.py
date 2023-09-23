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
