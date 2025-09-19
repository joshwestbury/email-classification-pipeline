"""
Email Taxonomy Discovery Pipeline

A modular pipeline for discovering sentiment and intent taxonomies from email datasets.
"""

from .config import PipelineConfig, ConfigManager
from .pipeline import TaxonomyPipeline

__version__ = "1.0.0"
__all__ = ["PipelineConfig", "ConfigManager", "TaxonomyPipeline"]