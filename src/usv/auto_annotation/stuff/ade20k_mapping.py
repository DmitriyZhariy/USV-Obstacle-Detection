"""
Static ADE20K class ID - (project_label, z_order) lookup table.

Only the subset of ADE20K classes relevant to USV maritime scenes is mapped.
All other ADE20K class IDs are intentionally absent — they produce no output.

Source Z-order values come from the annotation manual (configs/auto_annotation.yaml).

To extend: add rows here only. No Python code changes required.
"""
from __future__ import annotations

# ADE20K ID - (project label name, z_order)
ADE20K_TO_PROJECT: dict[int, tuple[str, int]] = {
    2:  ("Sky",     1),   # sky
    13: ("Land",    5),   # earth / ground
    1:  ("Land",    5),   # building - conservative fallback; annotator promotes to Pier
    21: ("Water",   0),   # water
    61: ("Bridge",  10),  # bridge
}

# Reverse lookup: project label - ADE20K IDs that map to it (for debugging / tests)
PROJECT_TO_ADE20K: dict[str, list[int]] = {}
for _ade_id, (_label, _z) in ADE20K_TO_PROJECT.items():
    PROJECT_TO_ADE20K.setdefault(_label, []).append(_ade_id)


def get_project_label(ade20k_id: int) -> tuple[str, int] | None:
    """
    Return (project_label, z_order) for an ADE20K class ID, or None if unmapped.

    Parameters
    ----------
    ade20k_id : int
        Raw class ID from SegFormer output (0-based ADE20K indexing).

    Returns
    -------
    tuple[str, int] | None
        (label, z_order) if the class is in the USV ontology, else None.
    """
    return ADE20K_TO_PROJECT.get(ade20k_id)
