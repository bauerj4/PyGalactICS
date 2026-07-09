"""Interface to the isolated legacy Fortran/C executables.

The original GalactICS implementation lives under ``legacy/fortran/`` at the
repository root. This subpackage provides path resolution and subprocess
runners; it does **not** import or re-export legacy source code.
"""

from galacticsics.legacy.paths import legacy_bin_dir, legacy_root, require_binary
from galacticsics.legacy.runner import LegacyRunner, LegacyRunError

__all__ = [
    "legacy_root",
    "legacy_bin_dir",
    "require_binary",
    "LegacyRunner",
    "LegacyRunError",
]
