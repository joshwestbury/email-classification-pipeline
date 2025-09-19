#!/usr/bin/env python3
"""
Clustering module for email taxonomy pipeline.

Performs UMAP dimensionality reduction and HDBSCAN clustering.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
import umap
import hdbscan
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class Clusterer:
    """Clusters email embeddings using UMAP + HDBSCAN."""

    def __init__(self, umap_params: Dict[str, Any] = None, hdbscan_params: Dict[str, Any] = None):
        self.umap_params = umap_params or {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 50,
            'random_state': 42
        }

        self.hdbscan_params = hdbscan_params or {
            'min_cluster_size': 5,
            'min_samples': 3,
            'metric': 'euclidean'
        }

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using UMAP."""
        logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {self.umap_params['n_components']}")

        reducer = umap.UMAP(**self.umap_params)
        reduced_embeddings = reducer.fit_transform(embeddings)

        logger.info(f"UMAP reduction complete: {reduced_embeddings.shape}")
        return reduced_embeddings

    def perform_clustering(self, reduced_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform HDBSCAN clustering."""
        logger.info("Performing HDBSCAN clustering...")

        clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)

        # Calculate statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0

        # Calculate silhouette score if we have clusters
        silhouette = None
        if n_clusters > 1:
            try:
                silhouette = silhouette_score(reduced_embeddings, cluster_labels)
            except ValueError:
                silhouette = None

        cluster_stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette_score': silhouette,
            'total_points': len(cluster_labels)
        }

        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.2%})")
        if silhouette is not None:
            logger.info(f"Silhouette score: {silhouette:.3f}")

        return cluster_labels, cluster_stats

    def analyze_clusters(self, cluster_labels: np.ndarray, metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cluster composition and characteristics."""
        cluster_analysis = {}

        # Group emails by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'index': idx,
                'metadata': metadata[idx] if idx < len(metadata) else {}
            })

        # Analyze each cluster
        for cluster_id, emails in clusters.items():
            cluster_size = len(emails)

            # Sample emails for analysis (max 10 per cluster)
            sample_emails = emails[:10] if cluster_size > 10 else emails

            cluster_info = {
                'cluster_id': int(cluster_id) if cluster_id != -1 else -1,
                'size': cluster_size,
                'percentage': (cluster_size / len(cluster_labels)) * 100,
                'sample_emails': [email['metadata'] for email in sample_emails],
                'sample_indices': [email['index'] for email in sample_emails]
            }

            cluster_analysis[str(cluster_id)] = cluster_info

        return cluster_analysis

    def cluster_emails(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform complete clustering pipeline."""
        logger.info("Starting clustering analysis...")

        # Use individual email embeddings for clustering
        embeddings = embeddings_data['individual']['embeddings']
        metadata = embeddings_data['individual']['metadata']

        if len(embeddings) == 0:
            logger.warning("No embeddings found for clustering")
            return {'error': 'No embeddings available'}

        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(embeddings)

        # Perform clustering
        cluster_labels, cluster_stats = self.perform_clustering(reduced_embeddings)

        # Analyze clusters
        cluster_analysis = self.analyze_clusters(cluster_labels, metadata)

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(
            [(k, v) for k, v in cluster_analysis.items() if k != '-1'],
            key=lambda x: x[1]['size'],
            reverse=True
        )

        # Add noise cluster at the end if it exists
        if '-1' in cluster_analysis:
            sorted_clusters.append(('-1', cluster_analysis['-1']))

        results = {
            'cluster_labels': cluster_labels.tolist(),
            'reduced_embeddings': reduced_embeddings.tolist(),
            'cluster_stats': cluster_stats,
            'cluster_analysis': dict(sorted_clusters),
            'clustering_config': {
                'umap_params': self.umap_params,
                'hdbscan_params': self.hdbscan_params
            },
            'top_clusters': [cluster_id for cluster_id, _ in sorted_clusters[:8] if cluster_id != '-1']
        }

        logger.info("Clustering analysis complete")
        return results

    def save_results(self, cluster_results: Dict[str, Any], output_path: str) -> None:
        """Save clustering results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved clustering results to {output_path}")