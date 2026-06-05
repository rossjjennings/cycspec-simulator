import numpy as np
import dask.array as da
import pytest
import os

import cycspec_simulator as cs

@pytest.fixture
def baseband_model():
    tests_dir = os.path.dirname(__file__)
    template_fname = os.path.join(
        tests_dir,
        "data/B1937+21.Rcvr1_2.GUPPI.15y.x.sum.sm"
    )
    template = cs.TemplateProfile.from_file(template_fname)
    template.normalize()
    template.make_posdef()
    polyco_fname = os.path.join(
        tests_dir,
        "data/polyco-B1937+21-60000.dat",
    )
    polyco = cs.PolynomialPredictor.from_file(polyco_fname)
    bandwidth = 1e6 # Hz
    obsfreq = 1.500e9 # Hz
    model = cs.BasebandModel(
        template=template,
        bandwidth=bandwidth,
        predictor=polyco,
        obsfreq=obsfreq,
    )
    return model

def test_add_filter(baseband_model):
    scattering_model = cs.ExponentialScatteringModel(
        scattering_time=40e-6, # s
        bandwidth=baseband_model.bandwidth,
        obsfreq=baseband_model.obsfreq,
        cutoff=20,
    )
    pattern = scattering_model.realize()
    baseband_model.add_filter(pattern)
    assert baseband_model.filters[-1] is pattern

def test_sample(baseband_model):
    data = baseband_model.sample(4096)
    assert isinstance(data, cs.BasebandData)
    assert np.abs(data.tspan - 0.004096) <= 1e-10
    assert data.A.shape == (4096,)
    assert data.B.shape == (4096,)

def test_sample_time(baseband_model):
    data = baseband_model.sample_time(0.005)
    assert isinstance(data, cs.BasebandData)
    assert np.abs(data.tspan - 0.005) <= 1e-10
    assert data.A.shape == (5000,)
    assert data.B.shape == (5000,)

def test_sample_rng(baseband_model):
    rng = np.random.default_rng()
    data = baseband_model.sample(4096, rng=rng)
    assert isinstance(data, cs.BasebandData)
    assert np.abs(data.tspan - 0.004096) <= 1e-10
    assert data.A.shape == (4096,)
    assert data.B.shape == (4096,)
    assert isinstance(data.A, np.ndarray)
    assert isinstance(data.B, np.ndarray)

def test_sample_dask(baseband_model):
    rng = da.random.default_rng()
    data = baseband_model.sample(4096, rng=rng)
    assert isinstance(data, cs.BasebandData)
    assert np.abs(data.tspan - 0.004096) <= 1e-10
    assert data.A.shape == (4096,)
    assert data.B.shape == (4096,)
    assert isinstance(data.A, da.Array)
    assert isinstance(data.B, da.Array)
