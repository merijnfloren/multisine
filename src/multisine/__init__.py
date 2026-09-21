"""Generate random-phase multisine excitation signals."""

from importlib.metadata import version

from ._multisine import (
    FrequencyInfo,
    RandomPhaseMultisine,
    random_phase_multisine,
    random_phase_orthogonal_multisine,
)

__all__ = [
    "FrequencyInfo",
    "RandomPhaseMultisine",
    "random_phase_multisine",
    "random_phase_orthogonal_multisine",
]

__version__ = version("multisine")
