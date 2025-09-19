#!/usr/bin/env python3
"""
Configuration management for the email taxonomy pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline run."""

    # Input data
    input_file: str
    dataset_name: str

    # Processing options
    clean_html: bool = True
    anonymize_pii: bool = True
    separate_threads: bool = True

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    include_thread_context: bool = True

    # Clustering parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 50
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3

    # LLM analysis
    openai_model: str = "gpt-4o"
    analyze_top_clusters: int = 8

    # Output settings
    output_dir: str = "outputs"
    save_intermediate: bool = True

    @classmethod
    def from_yaml(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def get_output_path(self, filename: str) -> Path:
        """Get output path for a file."""
        output_dir = Path(self.output_dir) / self.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename


class ConfigManager:
    """Manages pipeline configurations and templates."""

    @staticmethod
    def create_default_config(dataset_name: str, input_file: str) -> PipelineConfig:
        """Create a default configuration for a new dataset."""
        return PipelineConfig(
            dataset_name=dataset_name,
            input_file=input_file
        )

    @staticmethod
    def create_template_config(filepath: str) -> None:
        """Create a template configuration file."""
        template_config = PipelineConfig(
            dataset_name="example_dataset",
            input_file="path/to/your/emails.json"
        )
        template_config.to_yaml(filepath)