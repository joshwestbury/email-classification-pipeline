#!/usr/bin/env python3
"""
Simple test to verify DataProcessor runs without errors.
"""

import json
import logging
import sys
from pathlib import Path

# Add pipeline to Python path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from data_processor import DataProcessor

def test_data_processor():
    """Simple test to verify DataProcessor functionality."""

    # Setup minimal logging
    logging.basicConfig(level=logging.ERROR)  # Only show errors

    # Input and output paths
    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"
    output_file = "test_simple_output.json"

    print("Testing DataProcessor...")

    try:
        # Initialize data processor
        processor = DataProcessor(clean_html=True, separate_threads=True)

        # Process the emails
        processed_data = processor.process_emails(input_file)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        # Print basic summary
        metadata = processed_data['metadata']
        print(f"✓ SUCCESS: Processed {metadata['total_emails']} emails")
        print(f"✓ Threads: {metadata['total_threads']}")
        print(f"✓ Incoming: {metadata['incoming_emails']}")
        print(f"✓ Outgoing: {metadata['outgoing_emails']}")
        print(f"✓ Output saved to: {output_file}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_data_processor()
    if success:
        print("\n✓ DataProcessor test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ DataProcessor test failed!")
        sys.exit(1)