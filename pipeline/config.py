#!/usr/bin/env python3
"""
Configuration management for the email taxonomy pipeline.
"""

import yaml
import re
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
    embedding_model: str = "all-mpnet-base-v2"
    include_thread_context: bool = True

    # Clustering parameters - Granular settings for more specific categories
    umap_n_neighbors: int = 8        # Reduced from 15 for tighter neighborhoods
    umap_min_dist: float = 0.01      # Reduced from 0.1 for better separation
    umap_n_components: int = 35      # Reduced from 50 to avoid bottleneck
    hdbscan_min_cluster_size: int = 3  # Reduced from 5 for smaller clusters
    hdbscan_min_samples: int = 1     # Reduced from 3 for lower noise threshold

    # LLM analysis
    openai_model: str = "gpt-4o"
    analyze_top_clusters: int = 1000  # Analyze ALL clusters

    # Sentiment pre-analysis mode
    preanalysis_mode: str = "features"  # {"features", "labels"} - default "features"

    # System Prompt Generation
    generate_prompt: bool = True
    prompt_include_examples: bool = True
    prompt_confidence_scoring: bool = True
    prompt_entity_extraction: bool = True
    prompt_chain_of_thought: bool = True
    prompt_max_examples: int = 3

    # Output settings
    output_dir: str = "outputs"
    save_intermediate: bool = True

    @classmethod
    def from_yaml(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def load_granular_config(cls, input_file: str, dataset_name: str) -> 'PipelineConfig':
        """Load granular configuration optimized for specific categories."""
        granular_config_path = Path(__file__).parent / "granular_config.yaml"

        if granular_config_path.exists():
            # Load from granular config file and override input/dataset
            config = cls.from_yaml(str(granular_config_path))
            config.input_file = input_file
            config.dataset_name = dataset_name
            return config
        else:
            # Fallback to creating config with granular defaults (already set in class)
            return cls(input_file=input_file, dataset_name=dataset_name)

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
    def _get_next_analysis_number(output_dir: str = "outputs") -> int:
        """Get the next available analysis number for auto-naming."""
        output_path = Path(output_dir)

        if not output_path.exists():
            return 1

        # Find all existing directories with pattern "output_analysis_{number}"
        pattern = re.compile(r'^output_analysis_(\d+)$')
        max_number = 0

        for item in output_path.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    number = int(match.group(1))
                    max_number = max(max_number, number)

        return max_number + 1

    @staticmethod
    def create_default_config(dataset_name: str, input_file: str, auto_number: bool = True) -> PipelineConfig:
        """Create a default configuration for a new dataset."""
        if auto_number:
            # Generate auto-numbered dataset name
            next_number = ConfigManager._get_next_analysis_number()
            dataset_name = f"output_analysis_{next_number}"

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