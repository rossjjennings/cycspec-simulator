import numpy as np
import dask.array as da
import pytest
import os.path

from cycspec_simulator import PolynomialPredictor, Time

@pytest.fixture
def polyco():
    tests_dir = os.path.dirname(__file__)
    filename = os.path.join(tests_dir, "data/polyco-B1937+21-60000.dat")
    filename = os.path.normpath(filename)
    return PolynomialPredictor.from_file(filename)

def test_polyco_segments(polyco):
    assert len(polyco.segments) == 1

def test_polyco_covers(polyco):
    assert polyco.covers(Time(60000, 0, 0.))
    assert polyco.covers(Time(60000, 1800, 0.))
    span = 60 * polyco.segments[0].span # min -> sec
    sec, offs = np.modf(0.99*span)
    assert polyco.covers(Time(60000, int(sec), offs))

def test_polyco_phase(polyco):
    phase_epoch = polyco.phase(Time(60000, 0, 0.))
    phase_30min = polyco.phase(Time(60000, 1800, 0.))
    assert np.abs(phase_epoch - -1155516.122674652) <= 1e-8
    assert np.abs(phase_30min - 0.7849380372137075) <= 1e-8

def test_polyco_supports_array_time(polyco):
    offsets = np.linspace(0, 1800, 1025, endpoint=True)
    phase = polyco.phase(Time(60000, 0, offsets))
    assert phase.shape == (1025,)
    assert np.abs(phase[0] - -1155516.122674652) <= 1e-8
    assert np.abs(phase[-1] - 0.7849380372137075) <= 1e-8

def test_polyco_supports_dask(polyco):
    offsets = da.linspace(0, 1800, 1025, endpoint=True)
    phase = polyco.phase(Time(60000, 0, offsets))
    assert isinstance(phase, da.Array)
    assert phase.shape == (1025,)
    phase.compute()
    assert np.abs(phase[0] - -1155516.122674652) <= 1e-8
    assert np.abs(phase[-1] - 0.7849380372137075) <= 1e-8
