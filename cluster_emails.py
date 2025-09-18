#!/usr/bin/env python3
"""
Clustering Analysis for Collection Notes AI
Performs UMAP dimensionality reduction and HDBSCAN clustering on email embeddings
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
import umap
import hdbscan
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class EmailClusterer:
    """Handles clustering and visualization of email embeddings"""

    def __init__(self, embeddings_path: str, metadata_path: str):
        """Initialize clusterer with embedding and metadata paths"""
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.embeddings = None
        self.metadata = None
        self.umap_embeddings = None
        self.cluster_labels = None
        self.clusterer = None

    def load_data(self) -> None:
        """Load embeddings and metadata from files"""
        print("Loading embeddings and metadata...")

        # Load embeddings
        self.embeddings = np.load(self.embeddings_path)
        print(f"Loaded embeddings shape: {self.embeddings.shape}")

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded metadata for {len(self.metadata)} emails")

    def reduce_dimensions(self, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """Apply UMAP dimensionality reduction"""
        print(f"Applying UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}...")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
            verbose=True
        )

        self.umap_embeddings = reducer.fit_transform(self.embeddings)
        print(f"UMAP reduced embeddings shape: {self.umap_embeddings.shape}")

        return self.umap_embeddings

    def cluster_emails(self, min_cluster_size: int = 10, min_samples: int = 5) -> np.ndarray:
        """Apply HDBSCAN clustering"""
        print(f"Applying HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}...")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        self.cluster_labels = self.clusterer.fit_predict(self.umap_embeddings)

        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)

        print(f"Found {n_clusters} clusters with {n_noise} noise points")

        # Calculate silhouette score if we have clusters
        if n_clusters > 1:
            # Only calculate for non-noise points
            mask = self.cluster_labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(self.umap_embeddings[mask], self.cluster_labels[mask])
                print(f"Average silhouette score: {silhouette_avg:.3f}")

        return self.cluster_labels

    def visualize_clusters(self, save_plots: bool = True) -> None:
        """Create visualizations of the clusters"""
        if self.umap_embeddings is None or self.cluster_labels is None:
            print("Error: No embeddings or cluster labels available. Run clustering first.")
            return

        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'UMAP_1': self.umap_embeddings[:, 0],
            'UMAP_2': self.umap_embeddings[:, 1],
            'cluster': self.cluster_labels,
            'subject': [
                (email.get('subject') or '')[:50] + '...' if len(email.get('subject') or '') > 50
                else (email.get('subject') or '') for email in self.metadata
            ]
        })

        # Static plot with matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Colored by cluster
        scatter = ax1.scatter(df_plot['UMAP_1'], df_plot['UMAP_2'],
                             c=df_plot['cluster'], cmap='tab20', alpha=0.7, s=30)
        ax1.set_title('Email Clusters (UMAP + HDBSCAN)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('UMAP Dimension 1')
        ax1.set_ylabel('UMAP Dimension 2')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Cluster ID')

        # Plot 2: Highlight noise points
        noise_mask = df_plot['cluster'] == -1
        ax2.scatter(df_plot.loc[~noise_mask, 'UMAP_1'], df_plot.loc[~noise_mask, 'UMAP_2'],
                   c=df_plot.loc[~noise_mask, 'cluster'], cmap='tab20', alpha=0.7, s=30, label='Clustered')
        ax2.scatter(df_plot.loc[noise_mask, 'UMAP_1'], df_plot.loc[noise_mask, 'UMAP_2'],
                   c='black', alpha=0.5, s=20, label='Noise')
        ax2.set_title('Clustered vs Noise Points', fontsize=14, fontweight='bold')
        ax2.set_xlabel('UMAP Dimension 1')
        ax2.set_ylabel('UMAP Dimension 2')
        ax2.legend()

        plt.tight_layout()

        if save_plots:
            plt.savefig('email_clusters_static.png', dpi=300, bbox_inches='tight')
            print("Saved static cluster visualization as 'email_clusters_static.png'")

        plt.show()

        # Interactive plot with plotly
        fig_interactive = px.scatter(
            df_plot,
            x='UMAP_1',
            y='UMAP_2',
            color='cluster',
            hover_data=['subject'],
            title='Interactive Email Clusters (UMAP + HDBSCAN)',
            color_continuous_scale='viridis'
        )

        fig_interactive.update_traces(marker=dict(size=8, opacity=0.7))
        fig_interactive.update_layout(
            width=1000,
            height=700,
            title_font_size=16
        )

        if save_plots:
            fig_interactive.write_html('email_clusters_interactive.html')
            print("Saved interactive cluster visualization as 'email_clusters_interactive.html'")

        fig_interactive.show()

    def analyze_clusters(self) -> Dict[str, Any]:
        """Analyze cluster characteristics and return summary statistics"""
        if self.cluster_labels is None:
            print("Error: No cluster labels available. Run clustering first.")
            return {}

        cluster_stats = {}
        unique_clusters = np.unique(self.cluster_labels)

        print("\n=== CLUSTER ANALYSIS ===")

        for cluster_id in sorted(unique_clusters):
            if cluster_id == -1:
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster {cluster_id}"

            mask = self.cluster_labels == cluster_id
            cluster_emails = [self.metadata[i] for i in range(len(self.metadata)) if mask[i]]

            # Basic stats
            size = np.sum(mask)
            percentage = (size / len(self.cluster_labels)) * 100

            print(f"\n{cluster_name}:")
            print(f"  Size: {size} emails ({percentage:.1f}%)")

            # Sample subjects from this cluster
            subjects = [email.get('subject') or 'No Subject' for email in cluster_emails[:5]]
            print(f"  Sample subjects:")
            for i, subject in enumerate(subjects, 1):
                print(f"    {i}. {subject[:80]}{'...' if len(subject) > 80 else ''}")

            cluster_stats[str(cluster_id)] = {
                'size': size,
                'percentage': percentage,
                'sample_subjects': subjects,
                'emails': cluster_emails[:10]  # Store first 10 for detailed analysis
            }

        return cluster_stats

    def experiment_parameters(self) -> None:
        """Experiment with different clustering parameters"""
        print("\n=== PARAMETER EXPERIMENTATION ===")

        # UMAP parameters to try
        umap_configs = [
            {'n_neighbors': 10, 'min_dist': 0.05},
            {'n_neighbors': 15, 'min_dist': 0.1},
            {'n_neighbors': 20, 'min_dist': 0.2},
        ]

        # HDBSCAN parameters to try
        hdbscan_configs = [
            {'min_cluster_size': 5, 'min_samples': 3},
            {'min_cluster_size': 10, 'min_samples': 5},
            {'min_cluster_size': 20, 'min_samples': 10},
        ]

        results = []

        for umap_config in umap_configs:
            # Apply UMAP
            umap_embeddings = self.reduce_dimensions(**umap_config)

            for hdbscan_config in hdbscan_configs:
                # Apply HDBSCAN
                clusterer = hdbscan.HDBSCAN(**hdbscan_config)
                labels = clusterer.fit_predict(umap_embeddings)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)

                # Calculate silhouette score if possible
                silhouette = None
                if n_clusters > 1:
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        silhouette = silhouette_score(umap_embeddings[mask], labels[mask])

                result = {
                    'umap_neighbors': umap_config['n_neighbors'],
                    'umap_min_dist': umap_config['min_dist'],
                    'hdbscan_min_cluster': hdbscan_config['min_cluster_size'],
                    'hdbscan_min_samples': hdbscan_config['min_samples'],
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'silhouette': silhouette
                }

                results.append(result)

                print(f"UMAP({umap_config['n_neighbors']}, {umap_config['min_dist']}) + "
                      f"HDBSCAN({hdbscan_config['min_cluster_size']}, {hdbscan_config['min_samples']}): "
                      f"{n_clusters} clusters, {noise_ratio:.2f} noise ratio"
                      f"{f', silhouette: {silhouette:.3f}' if silhouette else ''}")

        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)

        print(f"\nBest configurations:")
        if len(results_df) > 0:
            # Best by number of clusters (moderate number)
            best_cluster_count = results_df.loc[(results_df['n_clusters'] >= 5) &
                                              (results_df['n_clusters'] <= 15)].nsmallest(1, 'noise_ratio')

            # Best by silhouette score
            valid_silhouette = results_df[results_df['silhouette'].notna()]
            if len(valid_silhouette) > 0:
                best_silhouette = valid_silhouette.nlargest(1, 'silhouette')

                print("Best by silhouette score:")
                print(best_silhouette.to_string(index=False))

            if len(best_cluster_count) > 0:
                print("\nBest by cluster count (5-15) and low noise:")
                print(best_cluster_count.to_string(index=False))

        return results_df


def main():
    """Main execution function"""
    print("=== EMAIL CLUSTERING ANALYSIS ===\n")

    # Initialize clusterer
    clusterer = EmailClusterer('email_embeddings.npy', 'email_metadata.json')

    # Load data
    clusterer.load_data()

    # Apply UMAP dimensionality reduction
    clusterer.reduce_dimensions(n_neighbors=15, min_dist=0.1)

    # Apply HDBSCAN clustering
    clusterer.cluster_emails(min_cluster_size=10, min_samples=5)

    # Visualize results
    clusterer.visualize_clusters(save_plots=True)

    # Analyze clusters
    cluster_stats = clusterer.analyze_clusters()

    # Save cluster results
    output_data = {
        'cluster_labels': clusterer.cluster_labels.tolist(),
        'cluster_stats': cluster_stats,
        'umap_embeddings': clusterer.umap_embeddings.tolist()
    }

    with open('cluster_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved cluster results to 'cluster_results.json'")

    # Experiment with different parameters
    print(f"\nRunning parameter experimentation...")
    results_df = clusterer.experiment_parameters()

    # Save parameter experiment results
    results_df.to_csv('parameter_experiments.csv', index=False)
    print(f"Saved parameter experiments to 'parameter_experiments.csv'")


if __name__ == "__main__":
    main()