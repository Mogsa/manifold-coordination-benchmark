"""
Atom registry — factory and parameter bank for ChaosBench v2.

Maps family names to atom classes, provides create_atom factory,
and stores the mini-bank parameter selections (non-textbook values).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

from chaosbench.grammar.atoms import (
    Atom,
    CircleAtom,
    DampedLinearAtom,
    HenonAtom,
    LogisticAtom,
    RotationAtom,
    SineAtom,
    TentAtom,
)


@dataclass(frozen=True)
class AtomSpec:
    """Specification for an atom family."""

    cls: Type[Atom]
    param_names: Tuple[str, ...]
    param_ranges: Dict[str, Tuple[float, float]]


ATOM_REGISTRY: Dict[str, AtomSpec] = {
    "logistic": AtomSpec(
        cls=LogisticAtom,
        param_names=("r",),
        param_ranges={"r": (2.5, 4.0)},
    ),
    "tent": AtomSpec(
        cls=TentAtom,
        param_names=("mu",),
        param_ranges={"mu": (1.0, 1.95)},
    ),
    "damped_linear": AtomSpec(
        cls=DampedLinearAtom,
        param_names=("lam",),
        param_ranges={"lam": (0.0, 0.99)},
    ),
    "rotation": AtomSpec(
        cls=RotationAtom,
        param_names=("omega",),
        param_ranges={"omega": (0.0, 1.0)},
    ),
    "sine": AtomSpec(
        cls=SineAtom,
        param_names=("a",),
        param_ranges={"a": (0.5, 1.0)},
    ),
    "circle": AtomSpec(
        cls=CircleAtom,
        param_names=("K", "omega"),
        param_ranges={"K": (0.0, 2.0), "omega": (0.0, 1.0)},
    ),
    "henon": AtomSpec(
        cls=HenonAtom,
        param_names=("a", "b"),
        param_ranges={"a": (1.0, 1.4), "b": (0.2, 0.4)},
    ),
}


def create_atom(family: str, params: dict) -> Atom:
    """Create an atom from family name and parameters."""
    if family not in ATOM_REGISTRY:
        raise ValueError(f"Unknown family '{family}'. Known: {list_families()}")
    spec = ATOM_REGISTRY[family]
    return spec.cls(**params)


def list_families() -> List[str]:
    """Return list of registered family names."""
    return list(ATOM_REGISTRY.keys())


# Non-textbook parameter selections for the mini-bank.
# Each family has 3 parameter sets covering different regimes.
# Critical: r=3.831 falls in period-3 window (λ=-0.37).
# Verified: r=3.891 is genuinely chaotic (λ≈0.49).
MINI_BANK_PARAMS = {
    "logistic": [
        {"r": 2.73},   # fixed_point (r < 3)
        {"r": 3.236},  # periodic (r ∈ [3, 3.57))
        {"r": 3.891},  # chaotic (λ≈0.49, verified NOT in periodic window)
    ],
    "tent": [
        {"mu": 1.237},  # chaotic, λ=0.21
        {"mu": 1.574},  # chaotic, λ=0.45
        {"mu": 1.891},  # chaotic, λ=0.64
    ],
    "damped_linear": [
        {"lam": 0.312},  # fast decay
        {"lam": 0.673},  # medium decay
        {"lam": 0.941},  # slow decay (near marginal)
    ],
    "rotation": [
        {"omega": 0.25},         # periodic (period 4)
        {"omega": 0.381966011},  # quasiperiodic (golden-ratio related)
        {"omega": 0.723},        # quasiperiodic
    ],
    "sine": [
        {"a": 0.65},  # periodic (below chaos onset)
        {"a": 0.84},  # near chaos onset
        {"a": 0.97},  # chaotic
    ],
    "circle": [
        {"K": 0.5, "omega": 0.382},   # quasiperiodic (K<1, irrational Ω)
        {"K": 0.9, "omega": 0.3},     # mode-locked periodic (Arnold tongue)
        {"K": 1.15, "omega": 0.618},  # chaotic (K>1)
    ],
    "henon": [
        {"a": 1.07, "b": 0.28},  # periodic
        {"a": 1.25, "b": 0.31},  # chaotic (λ≈0.27)
        {"a": 1.35, "b": 0.30},  # strongly chaotic (λ≈0.36)
    ],
}


# Conjugacy params: applied to each family's hardest param set (last entry).
# a=1.7, b=0.5 shifts domain so agent can't pattern-match on home domain.
CONJ_BANK_PARAMS = {
    "a": 1.7,
    "b": 0.5,
}
