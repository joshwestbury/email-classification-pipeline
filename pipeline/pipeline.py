#!/usr/bin/env python3
"""
Main pipeline orchestrator for email taxonomy discovery.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .config import PipelineConfig
from .data_processor import DataProcessor
from .anonymizer import Anonymizer
from .embedder import Embedder
from .clusterer import Clusterer
from .analyzer import LLMAnalyzer


class TaxonomyPipeline:
    """Main pipeline for email taxonomy discovery."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.state = {}  # Store intermediate results

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f'pipeline.{self.config.dataset_name}')

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete taxonomy discovery pipeline."""
        self.logger.info(f"Starting taxonomy pipeline for dataset: {self.config.dataset_name}")

        # Step 1: Data Processing
        if self.config.clean_html or self.config.separate_threads:
            self.logger.info("Step 1: Processing email data...")
            processed_data = self._run_data_processing()
            self.state['processed_data'] = processed_data

        # Step 2: PII Anonymization
        if self.config.anonymize_pii:
            self.logger.info("Step 2: Anonymizing PII...")
            anonymized_data = self._run_anonymization()
            self.state['anonymized_data'] = anonymized_data

        # Step 3: Embedding Generation
        self.logger.info("Step 3: Generating embeddings...")
        embeddings_data = self._run_embedding_generation()
        self.state['embeddings'] = embeddings_data

        # Step 4: Clustering
        self.logger.info("Step 4: Clustering emails...")
        cluster_results = self._run_clustering()
        self.state['clusters'] = cluster_results

        # Step 5: LLM Analysis
        self.logger.info("Step 5: Analyzing clusters with LLM...")
        analysis_results = self._run_llm_analysis()
        self.state['taxonomy'] = analysis_results

        self.logger.info("Pipeline completed successfully!")
        return self._generate_summary()

    def _run_data_processing(self) -> Dict[str, Any]:
        """Run data processing step."""
        processor = DataProcessor(
            clean_html=self.config.clean_html,
            separate_threads=self.config.separate_threads
        )

        processed_data = processor.process_emails(self.config.input_file)

        if self.config.save_intermediate:
            output_path = self.config.get_output_path('processed_emails.json')
            processor.save_results(processed_data, str(output_path))

        return processed_data

    def _run_anonymization(self) -> Dict[str, Any]:
        """Run PII anonymization step."""
        anonymizer = Anonymizer()

        # Use processed data if available, otherwise load from input
        if 'processed_data' in self.state:
            source_data = self.state['processed_data']
        else:
            with open(self.config.input_file, 'r') as f:
                source_data = json.load(f)

        anonymized_data = anonymizer.anonymize_dataset(source_data)

        if self.config.save_intermediate:
            output_path = self.config.get_output_path('anonymized_emails.json')
            anonymizer.save_results(anonymized_data, str(output_path))

        return anonymized_data

    def _run_embedding_generation(self) -> Dict[str, Any]:
        """Run embedding generation step."""
        embedder = Embedder(
            model_name=self.config.embedding_model,
            include_thread_context=self.config.include_thread_context
        )

        # Use most recent data from pipeline state
        if 'anonymized_data' in self.state:
            source_data = self.state['anonymized_data']
        elif 'processed_data' in self.state:
            source_data = self.state['processed_data']
        else:
            with open(self.config.input_file, 'r') as f:
                source_data = json.load(f)

        embeddings_data = embedder.generate_embeddings(source_data)

        if self.config.save_intermediate:
            embedder.save_embeddings(
                embeddings_data,
                self.config.get_output_path('embeddings')
            )

        return embeddings_data

    def _run_clustering(self) -> Dict[str, Any]:
        """Run clustering step."""
        clusterer = Clusterer(
            umap_params={
                'n_neighbors': self.config.umap_n_neighbors,
                'min_dist': self.config.umap_min_dist,
                'n_components': self.config.umap_n_components,
                'random_state': 42
            },
            hdbscan_params={
                'min_cluster_size': self.config.hdbscan_min_cluster_size,
                'min_samples': self.config.hdbscan_min_samples,
                'metric': 'euclidean'
            }
        )

        embeddings_data = self.state['embeddings']
        cluster_results = clusterer.cluster_emails(embeddings_data)

        if self.config.save_intermediate:
            output_path = self.config.get_output_path('cluster_results.json')
            clusterer.save_results(cluster_results, str(output_path))

        return cluster_results

    def _run_llm_analysis(self) -> Dict[str, Any]:
        """Run LLM analysis step."""
        analyzer = LLMAnalyzer(
            model=self.config.openai_model,
            top_clusters=self.config.analyze_top_clusters
        )

        cluster_results = self.state['clusters']

        # Get source data for analysis
        if 'anonymized_data' in self.state:
            source_data = self.state['anonymized_data']
        elif 'processed_data' in self.state:
            source_data = self.state['processed_data']
        else:
            with open(self.config.input_file, 'r') as f:
                source_data = json.load(f)

        analysis_results = analyzer.analyze_clusters(cluster_results, source_data)

        if self.config.save_intermediate:
            output_path = self.config.get_output_path('taxonomy_analysis.json')
            analyzer.save_results(analysis_results, str(output_path))

        return analysis_results

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the pipeline run."""
        summary = {
            'dataset': self.config.dataset_name,
            'config': self.config.__dict__,
            'results': {
                'total_emails': len(self.state.get('processed_data', {}).get('emails', [])),
                'clusters_found': len(self.state.get('clusters', {}).get('cluster_stats', {})),
                'categories_proposed': len(self.state.get('taxonomy', {}).get('categories', {}))
            },
            'output_files': [
                str(self.config.get_output_path(f)) for f in [
                    'processed_emails.json',
                    'anonymized_emails.json',
                    'embeddings',
                    'cluster_results.json',
                    'taxonomy_analysis.json'
                ]
            ]
        }

        # Save summary
        summary_path = self.config.get_output_path('pipeline_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary