import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(results_dir: Path) -> pd.DataFrame:
    summary_path = results_dir / "metrics_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find {summary_path}. Run the baseline script first."
        )

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    rows = []
    for task_name, metrics in summary.items():
        if task_name.startswith("dummy_"):
            continue

        if not isinstance(metrics, dict):
            continue

        row = {"task": task_name}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value
        rows.append(row)

    if not rows:
        raise ValueError("No numeric metrics found in metrics_summary.json.")

    df = pd.DataFrame(rows)

    df["task"] = df["task"].replace(
        {
            "binary": "Binary",
            "multiclass": "Multi-class",
        }
    )

    return df


def plot_metrics_with_values(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "roc_auc_ovr_macro",
    ]

    available_metrics = [m for m in metric_columns if m in metrics_df.columns]

    if not available_metrics:
        raise ValueError("No recognized metric columns found to plot.")

    display_names = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro F1",
        "roc_auc": "ROC-AUC",
        "roc_auc_ovr_macro": "ROC-AUC",
    }

    plot_df = metrics_df.set_index("task")[available_metrics]
    plot_df = plot_df.rename(columns=display_names)

    ax = plot_df.plot(kind="bar", figsize=(11, 6))

    ax.set_title("Baseline performance summary")
    ax.set_ylabel("Score")
    ax.set_xlabel("Classification task")
    ax.set_ylim(0, 1.12)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    for container in ax.containers:
        labels = [f"{bar.get_height():.3f}" for bar in container]
        ax.bar_label(
            container,
            labels=labels,
            label_type="edge",
            padding=3,
            fontsize=9,
            rotation=0,
        )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(csv_path: Path, output_path: Path, title: str) -> None:
    if not csv_path.exists():
        print(f"Skipping missing file: {csv_path}")
        return

    cm = pd.read_csv(csv_path, index_col=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=11,
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="resp_oximetry_baseline_results",
        help="Folder created by resp_oximetry_classical_baseline.py",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    metrics_df = load_metrics(results_dir)

    summary_table = metrics_df.copy()
    summary_table = summary_table.round(4)
    summary_csv = results_dir / "summary_table.csv"
    summary_table.to_csv(summary_csv, index=False)

    print("\n=== Baseline metrics summary ===")
    print(summary_table.to_string(index=False))

    metrics_plot = results_dir / "metrics_barplot_with_values.png"

    plot_metrics_with_values(metrics_df, metrics_plot)

    plot_confusion_matrix(
        results_dir / "confusion_matrix_binary.csv",
        results_dir / "confusion_matrix_binary.png",
        "Binary classification confusion matrix",
    )

    plot_confusion_matrix(
        results_dir / "confusion_matrix_multiclass.csv",
        results_dir / "confusion_matrix_multiclass.png",
        "Multi-class classification confusion matrix",
    )

    print("\nSaved files:")
    print(f"- {summary_csv}")
    print(f"- {metrics_plot}")
    print(f"- {results_dir / 'confusion_matrix_binary.png'}")
    print(f"- {results_dir / 'confusion_matrix_multiclass.png'}")


if __name__ == "__main__":
    main()
