"""Ask Physics: grounded, unit-checked answers to physics questions."""

from askphysics.mps_env import set_mps_watermark_defaults

# Before anything else in the package imports torch: torch reads these from the environment.
set_mps_watermark_defaults()

__version__ = "0.3.0"

__all__ = ["__version__"]
