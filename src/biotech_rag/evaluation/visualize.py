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
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

def _format_label(name: str) -> str:
    return name.replace("_", " ").replace("(mode=f1)", "(F1)")


def plot_rag_metrics(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot professional RAG metrics: radar (means) + per-sample distributions.

    Args:
        df: DataFrame containing metric columns (values in [0,1]).
        metrics: Optional list controlling metric order. If None, a sensible
            default order is used.
        figsize: Figure size.
        cmap: Seaborn palette name for the distribution plot.
        save_path: Optional path to save the rendered figure (PNG).
        show: Whether to call ``plt.show()`` before returning the Figure.

    Returns:
        The Matplotlib Figure object (also shown when ``show=True``).
    """
    default_metrics = [
        "context_recall",
        "answer_relevancy",
        "faithfulness",
        "factual_correctness(mode=f1)",
        "semantic_similarity",
        "context_precision",
    ]
    metrics = metrics or default_metrics

    present = [m for m in metrics if m in df.columns]
    missing = [m for m in metrics if m not in present]
    if missing:
        logger.warning("Missing metric columns: %s", missing)
    if not present:
        raise ValueError("No metric columns found in DataFrame.")

    # Numeric conversion and clipping to [0,1]
    df_num = df[present].apply(pd.to_numeric, errors="coerce").clip(0.0, 1.0)
    mean_vals = df_num.mean().reindex(present).fillna(0.0).values

    # Radar plot data
    N = len(present)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_loop = np.concatenate([angles, [angles[0]]])
    mean_loop = np.concatenate([mean_vals, [mean_vals[0]]])

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Left: radar (mean)
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.plot(angles_loop, mean_loop, color="#2E86AB", linewidth=2)
    ax1.fill(angles_loop, mean_loop, color="#2E86AB", alpha=0.25)
    labels = [_format_label(m) for m in present]
    ax1.set_thetagrids(np.degrees(angles), labels)
    ax1.set_ylim(0, 1)
    ax1.set_title("Mean RAG Metrics (0–1)", y=1.08, fontsize=12)

    # Right: per-sample distribution (box + jitter)
    ax2 = fig.add_subplot(1, 2, 2)
    df_long = df_num.melt(var_name="metric", value_name="value")
    order = present
    palette = sns.color_palette(cmap, n_colors=len(order))

    sns.barplot(x="value", y="metric", data=df_long, order=order, ax=ax2, palette=palette, hue="metric", errorbar=None)
    ax2.set_xlim(-0.02, 1.08)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("")
    ax2.set_title("Per-sample Metric Distribution", fontsize=12)

    # Add mean markers + numeric labels on the distribution plot
    yticks = ax2.get_yticks()
    # If seaborn created non-integer yticks, fall back to sequential positions
    if len(yticks) < len(mean_vals):
        yticks = np.arange(len(mean_vals))

    fig.suptitle("RAG Evaluation Metrics", fontsize=14)

    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved RAG metrics plot to %s", save_path)
        except Exception:
            logger.exception("Failed to save plot to %s", save_path)

    if show:
        plt.show()

    return None



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
