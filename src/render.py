"""Small rendering helpers used by the notebook.

Kept here so the notebook cells read like prose and don't drown in
matplotlib boilerplate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap(
    scores: np.ndarray,
    profile_names: list[str],
    jd_titles: list[str],
    *,
    title: str = "Cosine similarity: profiles x JDs",
) -> None:
    """Render the full N x M cosine matrix as an annotated heatmap."""
    df = pd.DataFrame(scores, index=profile_names, columns=jd_titles)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(jd_titles)),
                                    max(3, 0.6 * len(profile_names))))
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="viridis",
        cbar_kws={"label": "cosine similarity"},
        ax=ax, vmin=max(0.0, float(df.values.min())),
        vmax=min(1.0, float(df.values.max())),
    )
    ax.set_title(title)
    ax.set_xlabel("Job description")
    ax.set_ylabel("Candidate profile")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def ranked_list_per_jd(
    scores: np.ndarray,
    profile_names: list[str],
    jd_titles: list[str],
) -> pd.DataFrame:
    """One column per JD, values are profiles ranked best-first with score."""
    ranked = {}
    for j, jd in enumerate(jd_titles):
        col = scores[:, j]
        order = np.argsort(-col)
        ranked[jd] = [f"{profile_names[i]} ({col[i]:.2f})" for i in order]
    return pd.DataFrame(ranked)


def side_by_side(
    cosine_only: pd.DataFrame,
    filtered: pd.DataFrame,
    query: str,
) -> None:
    """Print the two rankings for the constrained query side by side."""
    print(f"Query: {query!r}\n")
    print("=" * 78)
    print("Ranking A — cosine only (no structured filter)")
    print("-" * 78)
    print(cosine_only.to_string(index=False))
    print()
    print("=" * 78)
    print("Ranking B — structured filter first, then cosine on survivors")
    print("-" * 78)
    if filtered.empty:
        print("(no profiles survived the structured filter)")
    else:
        print(filtered.to_string(index=False))
    print()
