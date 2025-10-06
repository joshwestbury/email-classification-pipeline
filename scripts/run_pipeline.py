#!/usr/bin/env python3
"""
CLI entry point for the email taxonomy discovery pipeline.

Usage examples:
    # Run with default raw_data directory (loads all JSON files)
    python run_pipeline.py
    python run_pipeline.py --dataset-name my_analysis

    # Run with specific input file
    python run_pipeline.py --input emails.json --dataset-name my_dataset

    # Run with specific directory
    python run_pipeline.py --input my_data_dir --dataset-name my_dataset

    # Create a template config
    python run_pipeline.py --create-template

    # Run with config file
    python run_pipeline.py --config my_dataset.yaml
"""

import argparse
import sys
import signal
import os
import atexit
from pathlib import Path

# Add project root to Python path so imports work from scripts/ directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def run_pipeline_quick(input_path: str | None = None, dataset_name: str | None = None) -> None:
    """Run the pipeline with granular configuration for specific categories."""
    # Default to raw_data directory if no input provided
    if input_path is None:
        input_path = "raw_data"
        print(f"No input specified - defaulting to raw_data directory")

    input_file = Path(input_path)

    if not input_file.exists():
        print(f"Error: Input path {input_file} not found.")
        sys.exit(1)

    # Determine if it's a directory or file
    if input_file.is_dir():
        print(f"Input is a directory - will load and concatenate all JSON files from {input_file}")
    else:
        print(f"Input is a file - will load {input_file}")

    # Use auto-numbering if no dataset name provided
    auto_number = dataset_name is None
    if dataset_name is None:
        dataset_name = "temp"  # Will be replaced by auto-numbering

    # Use granular configuration by default for better category specificity
    config = PipelineConfig.load_granular_config(str(input_file), dataset_name)

    # Apply auto-numbering if needed
    if auto_number:
        config = ConfigManager.apply_auto_numbering(config)

    pipeline = TaxonomyPipeline(config)

    results = pipeline.run_full_pipeline()
    print("\n=== PIPELINE COMPLETED ===")
    print(f"Dataset: {results['dataset']}")
    print(f"Total emails processed: {results['results']['total_emails']}")
    print(f"Clusters found: {results['results']['clusters_found']}")
    print(f"Categories proposed: {results['results']['categories_proposed']}")
    print(f"Output directory: {Path(config.output_dir) / config.dataset_name}")


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up and exiting...")
        # Force exit to ensure process terminates
        os._exit(0)

    def cleanup_on_exit():
        print("\n=== PIPELINE PROCESS TERMINATED ===")
        # Ensure we exit cleanly
        sys.stdout.flush()
        sys.stderr.flush()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination

    # Register cleanup function
    atexit.register(cleanup_on_exit)


def main():
    """Main CLI entry point."""
    # Set up signal handling for proper cleanup
    setup_signal_handlers()

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

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        "--create-template",
        action="store_true",
        help="Create a template configuration file",
    )

    group.add_argument("--config", type=str, help="Path to configuration YAML file")

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input path (file or directory). Defaults to 'raw_data' directory if not specified. If directory, all JSON files will be concatenated."
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

    try:
        if args.create_template:
            create_template_config(args.template_output)
        elif args.config:
            run_pipeline_from_config(args.config, args.steps)
        else:
            # Default to quick run (with or without --input)
            run_pipeline_quick(args.input, args.dataset_name)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user. Exiting...")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        sys.exit(1)
    finally:
        # Ensure we exit cleanly
        print("\n=== PIPELINE EXECUTION FINISHED ===")
        sys.exit(0)


if __name__ == "__main__":
    main()

