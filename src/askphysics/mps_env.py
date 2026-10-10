"""Memory limits for torch's MPS allocator, set before torch is imported.

torch reads ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` and ``PYTORCH_MPS_LOW_WATERMARK_RATIO`` from
the environment. The high ratio is the share of the GPU's working set that cached buffers may
fill before torch frees them, and the low ratio is where it stops freeing. Without them torch
keeps every buffer it has ever cached, and training at many widths fills the Mac's memory
until an allocation fails. ``askphysics/__init__.py`` calls ``set_mps_watermark_defaults``
before anything else imports torch, so every entry point gets the defaults. A value the
user set in the environment is never changed.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

HIGH_RATIO = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
LOW_RATIO = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
DEFAULT_HIGH = 0.7
DEFAULT_LOW = 0.5


def set_mps_watermark_defaults(environ: MutableMapping[str, str] = os.environ) -> None:
    """Set the MPS watermark ratios unless the environment already sets them.

    The low ratio must not exceed the high one. If only the high ratio is set, and it is
    below the default low ratio, the low ratio follows it down.
    """
    environ.setdefault(HIGH_RATIO, str(DEFAULT_HIGH))
    if LOW_RATIO not in environ:
        try:
            high = float(environ[HIGH_RATIO])
        except ValueError:
            high = DEFAULT_HIGH  # torch will reject the value itself; keep the default low
        environ[LOW_RATIO] = str(min(DEFAULT_LOW, high))
