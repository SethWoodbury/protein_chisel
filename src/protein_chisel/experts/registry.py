"""Expert registry: name -> :class:`Expert` factory.

Add a new model by writing an :class:`~protein_chisel.experts.base.Expert`
subclass and registering its factory in ``_FACTORIES`` (or via
:func:`register_expert`). Selection is by name, e.g.
``resolve_experts(["esmc", "saprot"], model_names={...})``.

Pattern borrowed from Sebastian (sebols) / Joe Mi's ``fused_mpnn_poe``
``get_expert`` registry.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from protein_chisel.experts.base import Expert
from protein_chisel.experts.esmc import ESMCExpert
from protein_chisel.experts.saprot import SaProtExpert

# Factories take an optional ``model_name`` (+ future kwargs) and return an Expert.
# Defaults match the legacy precompute defaults so behavior is unchanged.
_FACTORIES: dict[str, Callable[..., Expert]] = {
    "esmc": lambda model_name=None, **kw: ESMCExpert(model_name or "esmc_300m"),
    "saprot": lambda model_name=None, **kw: SaProtExpert(model_name or "saprot_35m"),
    # "hermes" registered in Phase 4 (experts/hermes.py).
}


def register_expert(name: str, factory: Callable[..., Expert]) -> None:
    """Register (or override) an expert factory under ``name``."""
    _FACTORIES[name.strip().lower()] = factory


def available_experts() -> list[str]:
    """Sorted list of registered expert names."""
    return sorted(_FACTORIES)


def get_expert(name: str, *, model_name: Optional[str] = None, **kw) -> Expert:
    """Instantiate one expert by name (raises ``KeyError`` if unknown)."""
    key = name.strip().lower()
    if key not in _FACTORIES:
        raise KeyError(
            f"unknown expert '{name}'; available: {available_experts()}"
        )
    return _FACTORIES[key](model_name=model_name, **kw)


def _parse_names(names: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(names, str):
        names = names.split(",")
    return [n.strip().lower() for n in names if str(n).strip()]


def resolve_experts(
    names: Union[str, Sequence[str]],
    *,
    model_names: Optional[dict[str, str]] = None,
) -> list[Expert]:
    """Resolve a name list/CSV into ordered :class:`Expert` instances.

    ``model_names`` optionally maps an expert name to its model variant
    (e.g. ``{"esmc": "esmc_600m", "saprot": "saprot_1.3b"}``).
    """
    model_names = model_names or {}
    out: list[Expert] = []
    for n in _parse_names(names):
        out.append(get_expert(n, model_name=model_names.get(n)))
    if not out:
        raise ValueError(f"no experts resolved from {names!r}")
    return out
