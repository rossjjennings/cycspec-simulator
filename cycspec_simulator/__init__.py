from .template_profile import TemplateProfile
from .baseband import BasebandModel, BasebandData
from .phase_predictor import FreqOnlyPredictor, PolynomialPredictor
from .scattering import ExponentialScatteringModel
from .interpolation import fft_roll, fft_interp, lerp
from .cycspec import cycfold_cpu
from .folding import fold
from .time import Time
from .metadata import ObservingMetadata

try:
    import cupy
    have_cuda = True
except ImportError:
    have_cuda = False
if have_cuda:
    from .cycspec_gpu import cycfold_gpu, cycfold_gpu_sharedmem

from . import _version
__version__ = _version.get_versions()['version']
