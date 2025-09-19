"""
Email Taxonomy Discovery Pipeline

A modular pipeline for discovering sentiment and intent taxonomies from email datasets.
"""

from .config import PipelineConfig, ConfigManager
from .pipeline import TaxonomyPipeline
from .data_processor import DataProcessor
from .anonymizer import Anonymizer
from .embedder import Embedder
from .clusterer import Clusterer
from .analyzer import LLMAnalyzer

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "ConfigManager",
    "TaxonomyPipeline",
    "DataProcessor",
    "Anonymizer",
    "Embedder",
    "Clusterer",
    "LLMAnalyzer"
]