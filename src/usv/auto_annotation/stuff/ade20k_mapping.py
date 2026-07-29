"""
Static ADE20K class ID - (project_label, class_id, z_order) lookup table.

Only the subset of ADE20K classes relevant to USV maritime scenes is mapped.
All other ADE20K class IDs are intentionally absent - they produce no output.

class_id and z_order values match configs/auto_annotation.yaml exactly.
This table must be kept in sync with the labels block in that config.

To extend: add rows here only. No Python code changes required elsewhere.
"""
from __future__ import annotations

# ADE20K ID - (project_label, class_id, z_order)
# class_id is the unique integer from configs/auto_annotation.yaml labels[].id
# z_order  is from configs/auto_annotation.yaml labels[].z_order
ADE20K_TO_PROJECT: dict[int, tuple[str, int, int]] = {
    # Water
    21: ("Water",  0,  0),   # water
    26: ("Water",  0,  0),   # sea
    60: ("Water",  0,  0),   # river
    128: ("Water",  0,  0),   # lake
    46: ("Water",  0,  0),   # sand

    # Sky
    2:  ("Sky",    1,  1),   # sky

    # Land
    13: ("Land",   2,  5),   # earth
    4:  ("Land",   2,  5),   # tree
    9:  ("Land",   2,  5),   # grass
    1:  ("Land",   2,  5),   # building
    16: ("Land",   2,  5),   # mountain
    # 140: ("Land",   2,  5),  # pier (станет Pier после ручной разметки)

    # Bridge
    61: ("Bridge", 4, 10),   # bridge

    # Pier
    140: ("Pier", 3, 10),  # pier
}
# NOTE: Pier (id=3, z=10) is NOT auto-generated - it requires human context
# to distinguish from building/Land. Annotators promote Land-Pier manually.

# Reverse lookup: project label - list of ADE20K IDs that map to it
# Useful for debugging and tests.
PROJECT_TO_ADE20K: dict[str, list[int]] = {}
for _ade_id, (_label, _cid, _z) in ADE20K_TO_PROJECT.items():
    PROJECT_TO_ADE20K.setdefault(_label, []).append(_ade_id)


def get_project_label(ade20k_id: int) -> tuple[str, int, int] | None:
    """
    Return (project_label, class_id, z_order) for an ADE20K class ID.

    Returns None if the class is not in the USV ontology.

    Parameters
    ----------
    ade20k_id : int
        Raw class ID from SegFormer output (0-based ADE20K indexing).

    Returns
    -------
    tuple[str, int, int] | None
        (label, class_id, z_order) or None if unmapped.
    """
    return ADE20K_TO_PROJECT.get(ade20k_id)
