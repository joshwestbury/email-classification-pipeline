#!/usr/bin/env python3
"""
Test the full pipeline with a subset of data
"""

import json
from pathlib import Path
from pipeline.config import PipelineConfig
from pipeline.pipeline import TaxonomyPipeline

def create_sample_data():
    """Create a sample of the original data for testing."""
    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"

    if not Path(input_file).exists():
        print(f"File {input_file} not found")
        return None

    # Read the first 50 records for testing
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Try standard JSON parsing first
    try:
        data = json.loads(content)
        sample_data = data[:50]
    except json.JSONDecodeError:
        # Use our malformed JSON parser
        from pipeline.data_processor import DataProcessor
        processor = DataProcessor()
        all_data = processor.parse_malformed_json(content)
        sample_data = all_data[:50]

    # Save sample data
    sample_file = "sample_collection_notes.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Created sample file with {len(sample_data)} emails: {sample_file}")
    return sample_file

def test_pipeline():
    """Test the full pipeline on sample data."""

    # Create sample data
    sample_file = create_sample_data()
    if not sample_file:
        return

    # Configure pipeline
    config = PipelineConfig(
        input_file=sample_file,
        dataset_name="sample_test",
        clean_html=True,
        separate_threads=True,
        save_intermediate=True
    )

    # Run pipeline steps individually to catch any errors
    pipeline = TaxonomyPipeline(config)

    try:
        print("Step 1: Processing data...")
        processed_data = pipeline._run_data_processing()
        print(f"✓ Processed {processed_data['metadata']['total_emails']} emails")
        print(f"  - Incoming: {processed_data['metadata']['incoming_emails']}")
        print(f"  - Outgoing: {processed_data['metadata']['outgoing_emails']}")

        print("\nStep 2: Anonymizing data...")
        anonymized_data = pipeline._run_anonymization()
        print(f"✓ Anonymized {len(anonymized_data['emails'])} emails")

        print("\nStep 3: Generating embeddings...")
        embeddings_data = pipeline._run_embedding_generation()
        print(f"✓ Generated embeddings: {embeddings_data['embeddings'].shape}")

        print("\nStep 4: Clustering...")
        cluster_results = pipeline._run_clustering()
        print(f"✓ Found {len(cluster_results['cluster_stats'])} clusters")

        print("\n✓ Pipeline test completed successfully!")

        # Show some sample results
        print("\nSample processed emails:")
        for i, email in enumerate(processed_data['emails'][:3]):
            print(f"\nEmail {i+1}:")
            print(f"  ID: {email['id']}")
            print(f"  Direction: {email['direction']}")
            print(f"  Thread Separated: {email.get('is_thread_separated', False)}")
            print(f"  Content: {email['content'][:100]}...")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()