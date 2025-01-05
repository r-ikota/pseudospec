from importlib import metadata
from .speclib import SpecCalc
from .speceq import SpecEQ

__version__ = metadata.version(__package__ or __name__)
