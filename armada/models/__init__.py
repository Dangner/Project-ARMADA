"""Model modules: Threat Profiler, grouped encoder, heads, GRL, discriminators,
TTT, memory and decision engine (spec §3)."""

from .core import ArmadaCore, ArmadaDomainModel
from .encoder import GroupedSelfAttentionEncoder, PreLNEncoderLayer
from .heads import ClassifierHead, balanced_pos_weight, build_pos_weight, classification_loss
from .profiler import ThreatProfiler

__all__ = [
    "ArmadaCore",
    "ArmadaDomainModel",
    "GroupedSelfAttentionEncoder",
    "PreLNEncoderLayer",
    "ClassifierHead",
    "ThreatProfiler",
    "balanced_pos_weight",
    "build_pos_weight",
    "classification_loss",
]
