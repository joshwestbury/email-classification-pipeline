#!/usr/bin/env python3
"""
CLI entry point for the email taxonomy discovery pipeline.

Usage examples:
    # Create a template config
    python run_pipeline.py --create-template

    # Run with config file
    python run_pipeline.py --config my_dataset.yaml

    # Quick run with minimal config
    python run_pipeline.py --input emails.json --dataset-name my_dataset

    # Run specific steps only
    python run_pipeline.py --config my_dataset.yaml --steps embedding,clustering,analysis
"""

import argparse
import sys
from pathlib import Path

from pipeline.config import ConfigManager, PipelineConfig
from pipeline.pipeline import TaxonomyPipeline


def create_template_config(output_path: str) -> None:
    """Create a template configuration file."""
    ConfigManager.create_template_config(output_path)
    print(f"Created template configuration at: {output_path}")
    print("Edit this file with your dataset details and rerun the pipeline.")


def run_pipeline_from_config(config_path: str, steps: str | None = None) -> None:
    """Run the pipeline using a configuration file."""
    if not Path(config_path).exists():
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)

    config = PipelineConfig.from_yaml(config_path)

    # Validate input file exists
    if not Path(config.input_file).exists():
        print(f"Error: Input file {config.input_file} not found.")
        sys.exit(1)

    pipeline = TaxonomyPipeline(config)

    if steps:
        print(f"Running specific steps: {steps}")
        # TODO: Implement step-specific execution
        print("Step-specific execution not yet implemented. Running full pipeline.")

    results = pipeline.run_full_pipeline()
    print("\n=== PIPELINE COMPLETED ===")
    print(f"Dataset: {results['dataset']}")
    print(f"Total emails processed: {results['results']['total_emails']}")
    print(f"Clusters found: {results['results']['clusters_found']}")
    print(f"Categories proposed: {results['results']['categories_proposed']}")
    print(f"Output directory: {Path(config.output_dir) / config.dataset_name}")


def run_pipeline_quick(input_file: str, dataset_name: str | None = None) -> None:
    """Run the pipeline with minimal configuration."""
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)

    # Use auto-numbering if no dataset name provided
    auto_number = dataset_name is None
    if dataset_name is None:
        dataset_name = "temp"  # Will be replaced by auto-numbering

    config = ConfigManager.create_default_config(
        dataset_name, input_file, auto_number=auto_number
    )
    pipeline = TaxonomyPipeline(config)

    results = pipeline.run_full_pipeline()
    print("\n=== PIPELINE COMPLETED ===")
    print(f"Dataset: {results['dataset']}")
    print(f"Total emails processed: {results['results']['total_emails']}")
    print(f"Clusters found: {results['results']['clusters_found']}")
    print(f"Categories proposed: {results['results']['categories_proposed']}")
    print(f"Output directory: {Path(config.output_dir) / config.dataset_name}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Email Taxonomy Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create template config
    python run_pipeline.py --create-template

    # Run with config file
    python run_pipeline.py --config dataset1.yaml

    # Quick run
    python run_pipeline.py --input emails.json --dataset-name dataset1
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--create-template",
        action="store_true",
        help="Create a template configuration file",
    )

    group.add_argument("--config", type=str, help="Path to configuration YAML file")

    group.add_argument(
        "--input", type=str, help="Input email JSON file (for quick run)"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name for the dataset (optional - will auto-generate output_analysis_# if not provided)",
    )

    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps to run (processing,anonymization,embedding,clustering,analysis)",
    )

    parser.add_argument(
        "--template-output",
        type=str,
        default="pipeline_config_template.yaml",
        help="Output path for template config (default: pipeline_config_template.yaml)",
    )

    args = parser.parse_args()

    if args.create_template:
        create_template_config(args.template_output)
    elif args.config:
        run_pipeline_from_config(args.config, args.steps)
    elif args.input:
        run_pipeline_quick(args.input, args.dataset_name)


if __name__ == "__main__":
    main()

