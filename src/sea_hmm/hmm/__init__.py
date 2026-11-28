"""Known-parameter hidden Markov processes."""

from ._process import StateEmittingHMM
from .processes import CpGIslandModel, GaussianRegimeSwitchingHMM, Mess3

__all__ = ["CpGIslandModel", "GaussianRegimeSwitchingHMM", "Mess3", "StateEmittingHMM"]
