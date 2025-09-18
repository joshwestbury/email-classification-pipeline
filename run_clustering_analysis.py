#!/usr/bin/env python3
"""
Run clustering analysis and parameter experimentation
"""

import json
import numpy as np
import pandas as pd
from cluster_emails import EmailClusterer

def main():
    """Run complete clustering analysis"""
    print("=== CLUSTERING ANALYSIS AND PARAMETER EXPERIMENTATION ===\n")

    # Initialize clusterer
    clusterer = EmailClusterer('incoming_email_embeddings.npy', 'incoming_email_metadata.json')

    # Load data
    clusterer.load_data()

    # Run parameter experimentation
    print("Running parameter experimentation...")
    results_df = clusterer.experiment_parameters()

    # Save results
    if len(results_df) > 0:
        results_df.to_csv('parameter_experiments.csv', index=False)
        print(f"Saved parameter experiments to 'parameter_experiments.csv'")

        # Show best results
        print("\n=== PARAMETER EXPERIMENT RESULTS ===")
        print(results_df.to_string(index=False))

        # Find optimal parameters
        valid_silhouette = results_df[results_df['silhouette'].notna()]
        if len(valid_silhouette) > 0:
            best_config = valid_silhouette.loc[valid_silhouette['silhouette'].idxmax()]
            print(f"\nBest configuration (highest silhouette score):")
            print(f"  UMAP: n_neighbors={best_config['umap_neighbors']}, min_dist={best_config['umap_min_dist']}")
            print(f"  HDBSCAN: min_cluster_size={best_config['hdbscan_min_cluster']}, min_samples={best_config['hdbscan_min_samples']}")
            print(f"  Results: {best_config['n_clusters']} clusters, {best_config['noise_ratio']:.2f} noise ratio, silhouette={best_config['silhouette']:.3f}")

    # Run analysis with default parameters and save cluster info
    print(f"\n=== RUNNING FINAL ANALYSIS WITH DEFAULT PARAMETERS ===")
    clusterer.reduce_dimensions()
    clusterer.cluster_emails()
    cluster_stats = clusterer.analyze_clusters()

    # Save cluster results with proper JSON serialization
    simple_stats = {}
    for cluster_id, stats in cluster_stats.items():
        simple_stats[str(cluster_id)] = {
            'size': int(stats['size']),
            'percentage': float(stats['percentage']),
            'sample_subjects': stats['sample_subjects']
        }

    output_data = {
        'n_clusters': len([k for k in simple_stats.keys() if k != '-1']),
        'n_noise': simple_stats.get('-1', {}).get('size', 0),
        'cluster_stats': simple_stats
    }

    with open('cluster_analysis_summary.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved cluster analysis summary to 'cluster_analysis_summary.json'")

if __name__ == "__main__":
    main()