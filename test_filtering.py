#!/usr/bin/env python3
"""
Test script to verify the thread filtering functionality.
"""

import json
import logging
import sys
from pathlib import Path

# Add pipeline to Python path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from data_processor import DataProcessor

def test_thread_filtering():
    """Test the thread filtering functionality."""

    # Setup minimal logging to show filtering info
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Input and output paths
    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"
    output_file = "test_filtered_output.json"

    print("Testing thread filtering functionality...")

    try:
        # Initialize data processor
        processor = DataProcessor(clean_html=True, separate_threads=True)

        # Process the emails
        processed_data = processor.process_emails(input_file)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        # Print summary
        metadata = processed_data['metadata']
        print(f"\n=== FILTERING RESULTS ===")
        print(f"Total threads before filtering: {metadata['total_threads']}")
        print(f"Relevant threads after filtering: {metadata['relevant_threads']}")
        print(f"Filtered out (single outgoing): {metadata['filtered_threads']}")
        print(f"Filtering reduction: {metadata['filtered_threads']/metadata['total_threads']*100:.1f}%")
        print(f"Total emails: {metadata['total_emails']}")
        print(f"Incoming emails: {metadata['incoming_emails']}")
        print(f"Outgoing emails: {metadata['outgoing_emails']}")
        print(f"Output saved to: {output_file}")

        # Show examples of remaining threads
        threads = processed_data['threads']
        print(f"\n=== SAMPLE RELEVANT THREADS ===")
        for i, thread in enumerate(threads[:5]):
            emails_in_thread = len(thread['emails'])
            incoming_in_thread = thread['metadata']['incoming_emails']
            outgoing_in_thread = thread['metadata']['outgoing_emails']
            subject = thread.get('subject', 'No subject')[:60]
            print(f"{i+1}. Thread {thread['thread_id']}: {emails_in_thread} emails "
                 f"({incoming_in_thread} incoming, {outgoing_in_thread} outgoing)")
            print(f"   Subject: {subject}...")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_thread_filtering()
    if success:
        print("\n✓ Thread filtering test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Thread filtering test failed!")
        sys.exit(1)