from importlib.metadata import version

from ._monitor import TMonitor as TMonitor
from .aliases import tenumerate as tenumerate
from .aliases import tmap as tmap
from .aliases import tproduct as tproduct
from .aliases import trange as trange
from .aliases import tzip as tzip
from .std import tldm as tldm

__version__ = version("tldm")
