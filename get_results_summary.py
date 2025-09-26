#!/usr/bin/env python3
"""
Results Summary Script

Usage: python get_results_summary.py results/fastvlm_ray_best2/
"""

import json
import os
import sys
import glob
from typing import Dict, Any


def load_metrics(results_dir: str) -> Dict[str, Dict[str, float]]:
    """Load metrics from all *_metrics.json files in the results directory."""
    metrics_files = glob.glob(os.path.join(results_dir, "*_metrics.json"))

    if not metrics_files:
        raise ValueError(f"No *_metrics.json files found in {results_dir}")

    # Load the main metrics file (should be the one without subdirectory)
    main_file = None
    for f in metrics_files:
        # Skip files in subdirectories
        if os.path.basename(os.path.dirname(f)) == os.path.basename(results_dir.rstrip('/')):
            main_file = f
            break

    if not main_file:
        main_file = metrics_files[0]  # fallback to first file

    print(f"Loading metrics from: {main_file}")

    with open(main_file, 'r') as f:
        data = json.load(f)

    return data['metrics']


def extract_key_metrics(benchmark_metrics: Dict[str, float]) -> Dict[str, float]:
    """Extract key metrics (ndcg@5, recall@5, mrr@5) from benchmark results."""
    return {
        'ndcg@5': benchmark_metrics.get('ndcg_at_5', 0.0),
        'recall@5': benchmark_metrics.get('recall_at_5', 0.0),
        'mrr@5': benchmark_metrics.get('mrr_at_5', 0.0)
    }


def clean_benchmark_name(name: str) -> str:
    """Clean benchmark name for display."""
    # Remove 'vidore/' prefix and common suffixes
    name = name.replace('vidore/', '')
    name = name.replace('_test_subsampled', '')
    name = name.replace('_test', '')
    return name


def print_results_table(metrics: Dict[str, Dict[str, float]]):
    """Print results in a formatted table."""

    # Extract key metrics for each benchmark
    results = {}
    for benchmark_name, benchmark_metrics in metrics.items():
        clean_name = clean_benchmark_name(benchmark_name)
        results[clean_name] = extract_key_metrics(benchmark_metrics)

    # Print header
    print("\n" + "="*80)
    print(f"{'Benchmark':<35} {'NDCG@5':<10} {'Recall@5':<12} {'MRR@5':<10}")
    print("="*80)

    # Print results for each benchmark
    totals = {'ndcg@5': 0, 'recall@5': 0, 'mrr@5': 0}
    count = 0

    for benchmark, metrics_dict in sorted(results.items()):
        ndcg = metrics_dict['ndcg@5']
        recall = metrics_dict['recall@5']
        mrr = metrics_dict['mrr@5']

        print(f"{benchmark:<35} {ndcg:<10.3f} {recall:<12.3f} {mrr:<10.3f}")

        totals['ndcg@5'] += ndcg
        totals['recall@5'] += recall
        totals['mrr@5'] += mrr
        count += 1

    # Print averages
    print("-"*80)
    avg_ndcg = totals['ndcg@5'] / count
    avg_recall = totals['recall@5'] / count
    avg_mrr = totals['mrr@5'] / count

    print(f"{'AVERAGE':<35} {avg_ndcg:<10.3f} {avg_recall:<12.3f} {avg_mrr:<10.3f}")
    print("="*80)
    print(f"\nSummary:")
    print(f"  • {count} benchmarks evaluated")
    print(f"  • Average NDCG@5: {avg_ndcg:.3f}")
    print(f"  • Average Recall@5: {avg_recall:.3f}")
    print(f"  • Average MRR@5: {avg_mrr:.3f}")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python get_results_summary.py <results_directory>")
        print("Example: python get_results_summary.py results/fastvlm_ray_best2/")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)

    try:
        metrics = load_metrics(results_dir)
        print_results_table(metrics)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()