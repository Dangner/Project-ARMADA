"""Feature-space adversarial attacks with EMBER constraints (spec §3.6).

All attacks act on the model's processed inputs and are projected onto the
EMBER-feasible region (non-negative count-like features, per-column source
max, frozen missingness).  They are NOT guaranteed to correspond to valid
PE executables — see README §Limitations.
"""

from .constraints import FeatureSpaceProjector
from .fgsm import fgsm_attack
from .pgd import pgd_attack

__all__ = ["FeatureSpaceProjector", "fgsm_attack", "pgd_attack"]
