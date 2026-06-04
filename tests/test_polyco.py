import numpy as np
import pytest
import os.path

from cycspec_simulator import PolynomialPredictor, Time

@pytest.fixture
def polyco():
    tests_dir = os.path.dirname(__file__)
    filename = os.path.join(tests_dir, "../examples/polyco-B1937+21-60000.dat")
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
