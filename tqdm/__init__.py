import os
from ._monitor import TMonitor, TqdmSynchronisationWarning
from .pandas import tqdm_pandas
from .std import (
    TqdmKeyError,
    TqdmTypeError, tqdm, trange)
from .utils import TqdmWarning
import importlib.metadata

if os.getenv('TQDM_NOTEBOOK'):
    from .notebook import tqdm, trange # noqa: F811


__all__ = ['tqdm', 'tqdm_gui', 'trange', 'tgrange', 'tqdm_pandas',
           'tqdm_notebook', 'tnrange', 'main', 'TMonitor',
           'TqdmTypeError', 'TqdmKeyError',
           'TqdmWarning', 'TqdmDeprecationWarning',
           'TqdmExperimentalWarning',
           'TqdmMonitorWarning', 'TqdmSynchronisationWarning',
           '__version__']


try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # Fallback for development mode or if the package is not installed yet
    __version__ = "0.0.0"
