"""
Sanity-check helpers.

Most graph checks are defined in `features.py`; this module re-exports them
and adds a compact import check.
"""
from .features import (
    zero_edge_check,
    summarize_graph_dataset,
    batch_sanity_check,
    check_paired_graph_alignment,
)


def package_import_check():
    """Return a small dict confirming the split package can be imported."""
    return {
        "package": "utils_tb",
        "status": "imported",
        "available_checks": [
            "zero_edge_check",
            "summarize_graph_dataset",
            "batch_sanity_check",
            "check_paired_graph_alignment",
        ],
    }
