"""Visualization utilities for retrieval evaluation.

This module provides a seaborn-preferred plotting function that
renders a grouped-bar chart, radar chart, and a summary table for
retrieval evaluation summaries (Precision@K, Recall@K, MRR).

Functions are deliberately lazy about importing heavy plotting
libraries so importing this module won't fail in minimal environments.

Example:
    from biotech_rag.evaluation.visualize import visualize_retrieval_summary
    visualize_retrieval_summary(result=result, out_dir="../data/processed")

"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def _coerce_summary(
    result: dict[str, Any] | None, summary_path: str | Path | None
) -> dict[str, Any]:
    """Load or normalize the evaluation summary.

    Args:
        result: In-memory evaluation dict (may include 'summary').
        summary_path: Path to a JSON summary file if `result` is None.

    Returns:
        A dict with a 'summary' key.
    """
    if result and isinstance(result, dict):
        if "summary" in result:
            return result
        # assume the dict is itself the summary
        return {"summary": result}

    if summary_path:
        summary_path = Path(summary_path)
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise ValueError("No evaluation result provided and summary_path not found.")


def visualize_retrieval_summary(
    result: dict[str, Any] | None = None,
    summary_path: str | Path | None = None,
    out_dir: str | Path = Path("../data/processed"),
    top_k: int = 5,
    figsize: tuple[int, int] = (18, 6),
    palette: Iterable[str] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Create seaborn-styled visualizations for retrieval evaluation.

    Produces a grouped bar chart (Precision/Recall/MRR), a radar chart
    (normalized metrics) and a small summary table. Saves a PNG to
    `out_dir` and returns paths and the plotting DataFrame.

    Args:
        result: Optional in-memory evaluation dict (as returned by
            `evaluate_retrieval`). If provided `summary_path` is ignored.
        summary_path: Optional path to a saved summary JSON file.
        out_dir: Directory where the visualization PNG will be saved.
        top_k: Top-K used during evaluation (used for title only).
        figsize: Figure size for the combined visualization.
        palette: Optional iterable of colors for the plotted methods.
        show: If True, display the figure inline (requires GUI/backends).

    Returns:
        Dict with keys: `fig_path` (Path to saved PNG) and `df` (pandas DataFrame used for plotting).
    """
    summary_container = _coerce_summary(result, summary_path)
    summary = summary_container["summary"]

    methods = ["vector", "hybrid", "hybrid_rerank"]
    display_map = {"vector": "Vector", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid+Rerank"}

    rows = []
    for m in methods:
        if m not in summary:
            continue
        rows.append(
            {
                "method": display_map.get(m, m),
                "precision": float(summary[m]["precision"]) * 100.0,
                "recall": float(summary[m]["recall"]) * 100.0,
                "mrr": float(summary[m]["mrr"]) * 100.0,
            }
        )

    if not rows:
        raise ValueError("No recognized methods found in summary.")

    df = pd.DataFrame(rows)
    df_rounded = df.copy()
    df_rounded[["precision", "recall", "mrr"]] = df_rounded[["precision", "recall", "mrr"]].round(2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import heavy plotting libraries
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError("Seaborn/matplotlib required for visualization: %s" % e)

    sns.set_theme(style="whitegrid", palette=palette or "muted", font_scale=1.05)

    # Single grouped barplot: x=metric (precision@K, recall@K, mrr), hue=method
    precision_label = f"precision@{top_k}"
    recall_label = f"recall@{top_k}"
    mrr_label = "mrr"

    df_plot = df.rename(
        columns={"precision": precision_label, "recall": recall_label, "mrr": mrr_label}
    )
    df_melt = df_plot.melt(
        id_vars="method",
        value_vars=[precision_label, recall_label, mrr_label],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(data=df_melt, x="metric", y="value", hue="method", ax=ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Percent (%)")
    ax.set_title(
        f"Retrieval Evaluation (n_queries={summary.get('n_queries', 'N/A')}, top_k={top_k})"
    )
    ax.legend(title="Method", loc="upper right")

    fig_path = out_dir / "retrieval_evaluation_viz.png"
    fig.savefig(str(fig_path), bbox_inches="tight", dpi=180)

    if show:
        try:
            plt.show()
        except Exception:
            pass

    # Return only the saved figure path (avoid returning large DataFrame or printing per-method tables)
    return str(fig_path)
