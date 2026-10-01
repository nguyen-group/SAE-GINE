"""
Source-code provenance search utilities for the final SLME implementation.
"""

from pathlib import Path

import pandas as pd


DEFAULT_SLME_PATTERNS = (
    "predicted_alpha_clean_cm1",
    "gap_cutoff_eV",
    "predicted_SLME_percent",
    "994.140",
    "Psolar",
    "P_SOLAR",
    "lambertw",
    "Lambert",
)


def find_exact_slme_implementation(
    project_root=r"D:\TB3",
    patterns=DEFAULT_SLME_PATTERNS,
    extensions=(".py", ".ipynb"),
    min_hits=2,
    max_files=20,
    max_context_hits=12,
    context_before=3,
    context_after=4,
    show=True,
):
    """
    Search the project tree for the most likely final SLME implementation.

    Candidate files are ranked by:
    1. Number of matched provenance patterns, descending.
    2. File path, ascending.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root directory to search recursively.
    patterns : sequence of str
        Text patterns associated with the final SLME implementation.
    extensions : sequence of str
        Source file extensions to search.
    min_hits : int
        Minimum number of distinct patterns required for a candidate file.
    max_files : int
        Maximum number of ranked candidates to print.
    max_context_hits : int
        Maximum number of matched line locations to expand per candidate.
    context_before, context_after : int
        Number of context lines printed around each matched line.
    show : bool
        Print the ranked source audit.

    Returns
    -------
    dict
        Ranked candidate table and detailed match metadata.
    """
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(
            f"Project root not found: {project_root}"
        )

    patterns = tuple(
        str(pattern)
        for pattern in patterns
    )

    normalized_patterns = tuple(
        pattern.lower()
        for pattern in patterns
    )

    normalized_extensions = tuple(
        extension
        if str(extension).startswith(".")
        else f".{extension}"
        for extension in extensions
    )

    matches = []

    for path in project_root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower()
            not in normalized_extensions
        ):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        text_lower = text.lower()

        hits = [
            pattern
            for pattern, normalized
            in zip(
                patterns,
                normalized_patterns,
            )
            if normalized in text_lower
        ]

        if len(hits) < int(min_hits):
            continue

        lines = text.splitlines()

        interesting_lines = []

        for line_index, line in enumerate(lines):
            line_lower = line.lower()

            if any(
                normalized in line_lower
                for normalized
                in normalized_patterns
            ):
                interesting_lines.append(
                    line_index
                )

        matches.append(
            {
                "path": path,
                "hits": hits,
                "hit_count": len(hits),
                "text": text,
                "lines": lines,
                "interesting_lines": interesting_lines,
            }
        )

    matches.sort(
        key=lambda item: (
            -item["hit_count"],
            str(item["path"]),
        )
    )

    candidate_rows = []

    for rank, item in enumerate(
        matches,
        start=1,
    ):
        candidate_rows.append(
            {
                "rank": rank,
                "path": str(item["path"]),
                "hit_count": item["hit_count"],
                "hits": " | ".join(item["hits"]),
                "matched_line_count": len(
                    item["interesting_lines"]
                ),
            }
        )

    candidates = pd.DataFrame(
        candidate_rows
    )

    if show:
        print("=" * 110)
        print("EXACT SLME SOURCE SEARCH")
        print("=" * 110)
        print(
            "Project root   :",
            project_root,
        )
        print(
            "Candidate files:",
            len(matches),
        )

        for rank, item in enumerate(
            matches[: int(max_files)],
            start=1,
        ):
            print()
            print("-" * 110)
            print(
                f"[{rank}] {item['path']}"
            )
            print(
                "Hits:",
                item["hits"],
            )

            shown_ranges = set()

            for line_index in item[
                "interesting_lines"
            ][: int(max_context_hits)]:
                start = max(
                    0,
                    line_index
                    - int(context_before),
                )

                end = min(
                    len(item["lines"]),
                    line_index
                    + int(context_after)
                    + 1,
                )

                key = (
                    start,
                    end,
                )

                if key in shown_ranges:
                    continue

                shown_ranges.add(
                    key
                )

                print(
                    f"\n--- lines "
                    f"{start + 1}-{end} ---"
                )

                for index in range(
                    start,
                    end,
                ):
                    print(
                        f"{index + 1:5d}: "
                        f"{item['lines'][index]}"
                    )

        print()
        print("=" * 110)
        print("SEARCH COMPLETE")
        print("=" * 110)

    return {
        "candidates": candidates,
        "matches": matches,
        "project_root": project_root,
        "patterns": patterns,
    }
