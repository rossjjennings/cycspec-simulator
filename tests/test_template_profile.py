import numpy as np
import matplotlib.pyplot as plt
import pytest
import os.path

from cycspec_simulator import TemplateProfile

@pytest.fixture
def template_file():
    tests_dir = os.path.dirname(__file__)
    filename = os.path.join(tests_dir, "../examples/B1937+21.Rcvr1_2.GUPPI.15y.x.sum.sm")
    return os.path.normpath(filename)

def template_from_scratch():
    phase = np.linspace(0, 1, 2048, endpoint=False)
    width = 0.01
    I = np.exp(-phase**2/(2*width**2))
    template = TemplateProfile(I)
    return template

def test_template_from_file(template_file):
    template = TemplateProfile.from_file(template_file)
    assert template.shape == (template.nbin,)
    assert template.full_stokes
    for attr in ['I', 'Q', 'U', 'V', 'phase']:
        assert getattr(template, attr).shape == template.shape
    assert template.nbin == 2048

def test_template_from_scratch():
    template = template_from_scratch()
    assert template.shape == (template.nbin,)
    assert not template.full_stokes
    for attr in ['I', 'phase']:
        assert getattr(template, attr).shape == template.shape
    assert template.nbin == 2048

def test_normalize(template_file):
    template = TemplateProfile.from_file(template_file)
    template.normalize()
    assert np.max(template.I) == 1.

def test_make_posdef(template_file):
    template = TemplateProfile.from_file(template_file)
    template.make_posdef()
    assert np.all(template.squared_norm) >= 0

def test_plot(template_file):
    template = TemplateProfile.from_file(template_file)
    template.make_posdef()
    template.plot(what='I', shift=0.25)
    template.plot(what='S', shift=0.25)
    template.plot(what='S2', shift=0.25)
    template.plot(what='ILV', shift=0.25)
    template.plot(what='IQUV', shift=0.25)

def test_resample(template_file):
    template = TemplateProfile.from_file(template_file)
    new_template = template.resample(200)
    assert new_template.shape == (new_template.nbin,)
    assert new_template.full_stokes
    for attr in ['I', 'Q', 'U', 'V', 'phase']:
        assert getattr(new_template, attr).shape == new_template.shape
    assert new_template.nbin == 200
