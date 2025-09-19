#!/usr/bin/env python3
"""
Test script to run the updated DataProcessor with thread-grouped output structure.
"""

import json
import logging
import sys
from pathlib import Path

# Add pipeline to Python path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from data_processor import DataProcessor

def test_thread_grouped_structure():
    """Test the updated thread-grouped structure functionality."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Input and output paths
    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"
    output_file = "test_thread_grouped_results.json"

    logger.info("Starting thread-grouped structure test...")

    # Initialize data processor
    processor = DataProcessor(clean_html=True, separate_threads=True)

    # Process the emails
    try:
        processed_data = processor.process_emails(input_file)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        # Print summary
        metadata = processed_data['metadata']
        threads = processed_data['threads']

        logger.info(f"Processing complete!")
        logger.info(f"Total threads: {metadata['total_threads']}")
        logger.info(f"Total emails: {metadata['total_emails']}")
        logger.info(f"Incoming emails: {metadata['incoming_emails']}")
        logger.info(f"Outgoing emails: {metadata['outgoing_emails']}")
        logger.info(f"Results saved to: {output_file}")

        # Show examples of thread structure
        logger.info(f"\nFirst 3 threads with their email counts:")
        for i, thread in enumerate(threads[:3]):
            emails_in_thread = len(thread['emails'])
            incoming_in_thread = thread['metadata']['incoming_emails']
            outgoing_in_thread = thread['metadata']['outgoing_emails']
            subject = thread.get('subject', 'No subject')[:60]
            logger.info(f"{i+1}. Thread {thread['thread_id']}: {emails_in_thread} emails "
                       f"({incoming_in_thread} incoming, {outgoing_in_thread} outgoing)")
            logger.info(f"   Subject: {subject}...")

            # Show first 2 emails in this thread
            for j, email in enumerate(thread['emails'][:2]):
                direction = email['direction'].upper()
                content = email['content'][:50].replace('\n', ' ')
                sender = email.get('sender', 'Unknown')[:20]
                logger.info(f"     {j+1}. [{direction}] {content}... (from: {sender})")
            logger.info("")

        return processed_data

    except Exception as e:
        logger.error(f"Error processing emails: {e}")
        raise

if __name__ == "__main__":
    test_thread_grouped_structure()