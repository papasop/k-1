"""Compatibility package for the legacy monolithic implementation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_LEGACY_MODULE_NAME = "_lorentz_transformer_legacy"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "lorentz_transformer.py"
_LEGACY_SPEC = spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)

if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:
    raise ImportError(f"Unable to load legacy module from {_LEGACY_PATH}")

_legacy_module = module_from_spec(_LEGACY_SPEC)
sys.modules.setdefault(_LEGACY_MODULE_NAME, _legacy_module)
_LEGACY_SPEC.loader.exec_module(_legacy_module)

__doc__ = _legacy_module.__doc__
__all__ = [name for name in dir(_legacy_module) if not name.startswith("_")]

globals().update(
    {name: getattr(_legacy_module, name) for name in __all__}
)
