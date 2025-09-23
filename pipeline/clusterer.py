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
from sklearn.cluster import AgglomerativeClustering
import logging

logger = logging.getLogger(__name__)


class Clusterer:
    """Clusters email embeddings using UMAP + HDBSCAN."""

    def __init__(self, umap_params: Dict[str, Any] = None, hdbscan_params: Dict[str, Any] = None, hierarchical_mode: bool = True):
        # Enhanced UMAP parameters for finer granularity
        self.umap_params = umap_params or {
            'n_neighbors': 10,  # Reduced from 15 for finer detail
            'min_dist': 0.05,   # Reduced from 0.1 for tighter clusters
            'n_components': 30, # Reduced from 50 for better clustering
            'random_state': 42
        }

        # Enhanced HDBSCAN parameters for multi-stage clustering
        self.hdbscan_params = hdbscan_params or {
            'min_cluster_size': 8,  # Broad intent categories
            'min_samples': 3,
            'metric': 'euclidean',
            'cluster_selection_epsilon': 0.3
        }

        self.hierarchical_mode = hierarchical_mode

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using UMAP."""
        logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {self.umap_params['n_components']}")

        reducer = umap.UMAP(**self.umap_params)
        reduced_embeddings = reducer.fit_transform(embeddings)

        logger.info(f"UMAP reduction complete: {reduced_embeddings.shape}")
        return reduced_embeddings

    def perform_hierarchical_clustering(self, reduced_embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform two-stage hierarchical clustering for intent and sentiment."""
        logger.info("Performing hierarchical clustering (Stage 1: Intent categories)")

        # Stage 1: Broad intent clustering with larger min_cluster_size
        stage1_params = self.hdbscan_params.copy()
        stage1_params['min_cluster_size'] = 8  # Broad intent categories

        stage1_clusterer = hdbscan.HDBSCAN(**stage1_params)
        intent_labels = stage1_clusterer.fit_predict(reduced_embeddings)

        # Track original indices for each intent cluster
        intent_clusters = {}
        for idx, label in enumerate(intent_labels):
            if label not in intent_clusters:
                intent_clusters[label] = []
            intent_clusters[label].append(idx)

        # Stage 2: Sentiment sub-clustering within each intent cluster
        logger.info("Performing hierarchical clustering (Stage 2: Sentiment sub-clusters)")

        final_labels = np.copy(intent_labels)
        cluster_counter = max(intent_labels) + 1 if len(intent_labels) > 0 else 0
        hierarchical_info = {}

        for intent_label, indices in intent_clusters.items():
            if intent_label == -1 or len(indices) < 6:  # Skip noise and small clusters
                continue

            # Extract embeddings for this intent cluster
            intent_embeddings = reduced_embeddings[indices]

            # Stage 2: Finer sentiment clustering within intent
            stage2_params = {
                'min_cluster_size': 3,  # Sentiment sub-clusters
                'min_samples': 2,
                'metric': 'euclidean'
            }

            stage2_clusterer = hdbscan.HDBSCAN(**stage2_params)
            sentiment_labels = stage2_clusterer.fit_predict(intent_embeddings)

            # Map sentiment sub-clusters back to global labels
            for local_idx, sentiment_label in enumerate(sentiment_labels):
                global_idx = indices[local_idx]
                if sentiment_label != -1:  # Valid sub-cluster
                    final_labels[global_idx] = cluster_counter + sentiment_label
                # else: keep original intent label

            # Track hierarchical structure
            n_sentiment_clusters = len(set(sentiment_labels)) - (1 if -1 in sentiment_labels else 0)
            hierarchical_info[str(intent_label)] = {
                'intent_size': int(len(indices)),
                'sentiment_subclusters': int(n_sentiment_clusters),
                'sentiment_labels': [int(x) for x in sentiment_labels.tolist()]
            }

            cluster_counter += max(sentiment_labels) + 1 if len(sentiment_labels) > 0 else 0

        # Calculate final statistics
        n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
        n_noise = list(final_labels).count(-1)
        noise_ratio = n_noise / len(final_labels) if len(final_labels) > 0 else 0

        # Calculate silhouette score
        silhouette = None
        if n_clusters > 1:
            try:
                silhouette = silhouette_score(reduced_embeddings, final_labels)
            except ValueError:
                silhouette = None

        cluster_stats = {
            'clustering_method': 'hierarchical',
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_ratio': float(noise_ratio),
            'silhouette_score': float(silhouette) if silhouette is not None else None,
            'total_points': int(len(final_labels)),
            'hierarchical_info': hierarchical_info,
            'stage1_intent_clusters': int(len(set(intent_labels)) - (1 if -1 in intent_labels else 0)),
            'stage2_total_subclusters': int(n_clusters)
        }

        logger.info(f"Hierarchical clustering complete: {cluster_stats['stage1_intent_clusters']} intent clusters → {n_clusters} total clusters")

        return final_labels, cluster_stats

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
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_ratio': float(noise_ratio),
            'silhouette_score': float(silhouette) if silhouette is not None else None,
            'total_points': int(len(cluster_labels))
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
                'size': int(cluster_size),
                'percentage': float((cluster_size / len(cluster_labels)) * 100),
                'sample_emails': [email['metadata'] for email in sample_emails],
                'sample_indices': [int(email['index']) for email in sample_emails]
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

        # Perform clustering (hierarchical or standard)
        if self.hierarchical_mode:
            cluster_labels, cluster_stats = self.perform_hierarchical_clustering(reduced_embeddings, metadata)
        else:
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