import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import wandb
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(path: str, obj: Any) -> str:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def _metric_direction(metric_name: str) -> str:
    m = metric_name.lower()
    if any(x in m for x in ["loss", "error", "perplexity", "kl"]) and "rouge" not in m:
        return "min"
    return "max"


def _plot_learning_curve(df: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths = []
    key_pairs = [
        ("train_loss", "val_loss"),
        ("val_rougeL", None),
        ("val_segment_coverage_kl", None),
        ("val_late_segment_mass", None),
    ]
    for kp in key_pairs:
        k1, k2 = kp
        if k1 not in df.columns and (k2 is None or k2 not in df.columns):
            continue

        plt.figure(figsize=(8, 4))
        if k1 in df.columns:
            plt.plot(df["_step"], df[k1], label=k1)
            if df[k1].notna().any():
                last = df[k1].dropna().iloc[-1]
                plt.annotate(f"{last:.4g}", (df["_step"].dropna().iloc[-1], last))
        if k2 is not None and k2 in df.columns:
            plt.plot(df["_step"], df[k2], label=k2)
            if df[k2].notna().any():
                last = df[k2].dropna().iloc[-1]
                plt.annotate(f"{last:.4g}", (df["_step"].dropna().iloc[-1], last))

        plt.xlabel("step")
        plt.ylabel("value")
        plt.title(f"{run_id}: {k1}" + (f" vs {k2}" if k2 else ""))
        plt.legend()
        plt.tight_layout()
        fname = f"{run_id}_learning_curve_{k1}" + (f"_{k2}" if k2 else "") + ".pdf"
        path = os.path.join(out_dir, fname)
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths


def _plot_bar(values: Dict[str, float], title: str, ylabel: str, out_path: str) -> str:
    plt.figure(figsize=(10, 4))
    run_ids = list(values.keys())
    vals = [values[r] for r in run_ids]
    sns.barplot(x=run_ids, y=vals)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.4g}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--run_ids", type=str, required=True)
    args = ap.parse_args()

    results_dir = args.results_dir
    run_ids: List[str] = json.loads(args.run_ids)

    cfg = OmegaConf.load(os.path.join("config", "config.yaml"))
    entity = str(cfg.wandb.entity)
    project = str(cfg.wandb.project)

    api = wandb.Api()

    per_run_summaries: Dict[str, Dict[str, Any]] = {}
    per_run_histories: Dict[str, pd.DataFrame] = {}
    generated_paths: List[str] = []

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        hist = run.history(pandas=True)
        summ = run.summary._json_dict
        conf = dict(run.config)

        out_dir = _ensure_dir(os.path.join(results_dir, run_id))
        metrics_path = os.path.join(out_dir, "metrics.json")
        exported = {
            "run_id": run_id,
            "config": conf,
            "summary": summ,
            "history_columns": list(hist.columns),
        }
        generated_paths.append(_save_json(metrics_path, exported))

        if "_step" not in hist.columns:
            hist["_step"] = np.arange(len(hist))

        fig_paths = _plot_learning_curve(hist, run_id=run_id, out_dir=out_dir)
        generated_paths.extend(fig_paths)

        per_run_summaries[run_id] = summ
        per_run_histories[run_id] = hist

    # Aggregated analysis
    comp_dir = _ensure_dir(os.path.join(results_dir, "comparison"))

    primary_metric = "rougeL"
    metrics: Dict[str, Dict[str, float]] = {}

    def _get_summary_metric(summary: Dict[str, Any], key: str) -> float:
        # allow either best_val_rougeL_mean or val_rougeL style
        for cand in [
            f"best_val_{key}_mean",
            f"best_val_{key}",
            f"val_{key}",
            key,
        ]:
            if cand in summary:
                try:
                    return float(summary[cand])
                except Exception:
                    continue
        return float("nan")

    all_metric_names = [
        "rougeL",
        "rouge1",
        "rouge2",
        "segment_coverage_kl",
        "late_segment_mass",
        "tokens_per_sec",
        "peak_gpu_mem",
        "best_val_rougeL_mean",
        "best_val_rougeL_std",
    ]

    for mn in all_metric_names:
        metrics[mn] = {}
        for run_id in run_ids:
            metrics[mn][run_id] = _get_summary_metric(per_run_summaries[run_id], mn)

    proposed_candidates = [(rid, _get_summary_metric(per_run_summaries[rid], primary_metric)) for rid in run_ids if "proposed" in rid]
    baseline_candidates = [(rid, _get_summary_metric(per_run_summaries[rid], primary_metric)) for rid in run_ids if ("baseline" in rid or "comparative" in rid)]

    best_proposed = max(proposed_candidates, key=lambda x: (-1e18 if np.isnan(x[1]) else x[1])) if proposed_candidates else (None, float("nan"))
    best_baseline = max(baseline_candidates, key=lambda x: (-1e18 if np.isnan(x[1]) else x[1])) if baseline_candidates else (None, float("nan"))

    direction = _metric_direction(primary_metric)
    gap = float("nan")
    if best_proposed[0] is not None and best_baseline[0] is not None and not np.isnan(best_proposed[1]) and not np.isnan(best_baseline[1]) and best_baseline[1] != 0:
        raw_gap = (best_proposed[1] - best_baseline[1]) / best_baseline[1] * 100.0
        gap = raw_gap if direction == "max" else -raw_gap

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": metrics,
        "best_proposed": {"run_id": best_proposed[0], "value": best_proposed[1]},
        "best_baseline": {"run_id": best_baseline[0], "value": best_baseline[1]},
        "gap": gap,
    }

    agg_path = os.path.join(comp_dir, "aggregated_metrics.json")
    generated_paths.append(_save_json(agg_path, aggregated))

    # Comparison figures
    primary_vals = {rid: _get_summary_metric(per_run_summaries[rid], primary_metric) for rid in run_ids}
    fig1 = os.path.join(comp_dir, f"comparison_{primary_metric}_bar_chart.pdf")
    generated_paths.append(_plot_bar(primary_vals, title=f"Comparison: {primary_metric}", ylabel=primary_metric, out_path=fig1))

    kl_vals = {rid: _get_summary_metric(per_run_summaries[rid], "segment_coverage_kl") for rid in run_ids}
    fig2 = os.path.join(comp_dir, "comparison_segment_coverage_kl_bar_chart.pdf")
    generated_paths.append(_plot_bar(kl_vals, title="Comparison: segment_coverage_kl (lower is better)", ylabel="segment_coverage_kl", out_path=fig2))

    # Significance test (paired by seed not available; use simple bootstrap over per-run val_rougeL points if present)
    # We attempt to use history val_rougeL points as samples.
    pval = None
    if best_proposed[0] is not None and best_baseline[0] is not None:
        hp = per_run_histories[best_proposed[0]]
        hb = per_run_histories[best_baseline[0]]
        if "val_rougeL" in hp.columns and "val_rougeL" in hb.columns:
            xp = hp["val_rougeL"].dropna().values
            xb = hb["val_rougeL"].dropna().values
            if len(xp) >= 5 and len(xb) >= 5:
                rng = np.random.default_rng(0)
                nboot = 2000
                diffs = []
                for _ in range(nboot):
                    sp = rng.choice(xp, size=len(xp), replace=True).mean()
                    sb = rng.choice(xb, size=len(xb), replace=True).mean()
                    diffs.append(sp - sb)
                diffs = np.array(diffs)
                pval = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))

    if pval is not None:
        p_path = os.path.join(comp_dir, "comparison_bootstrap_pvalue.json")
        generated_paths.append(_save_json(p_path, {"best_proposed": best_proposed[0], "best_baseline": best_baseline[0], "p_value": pval}))

    for p in generated_paths:
        print(p)


if __name__ == "__main__":
    main()
