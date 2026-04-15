from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cardiomas")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from cardiomas.api import CardioMAS

__all__ = ["CardioMAS", "__version__"]
