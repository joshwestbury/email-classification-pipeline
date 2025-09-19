#!/usr/bin/env python3
"""
Test script to run just the email parsing and categorization part of the pipeline.
"""

import json
import logging
import sys
from pathlib import Path

# Add pipeline to Python path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from data_processor import DataProcessor

def test_email_categorization():
    """Test the email parsing and categorization functionality."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Input and output paths
    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"
    output_file = "test_email_categorization_results.json"

    logger.info("Starting email categorization test...")

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
        logger.info(f"Processing complete!")
        logger.info(f"Total emails processed: {metadata['total_emails']}")
        logger.info(f"Incoming emails: {metadata['incoming_emails']}")
        logger.info(f"Outgoing emails: {metadata['outgoing_emails']}")
        logger.info(f"Results saved to: {output_file}")

        # Show a few examples
        emails = processed_data['emails']
        logger.info("\nFirst 5 categorized emails:")
        for i, email in enumerate(emails[:5]):
            direction = email.get('direction', 'unknown')
            subject = email.get('subject', 'No subject')[:50]
            sender = email.get('sender', 'Unknown sender')[:30]
            logger.info(f"{i+1}. [{direction.upper()}] {subject}... (from: {sender})")

        return processed_data

    except Exception as e:
        logger.error(f"Error processing emails: {e}")
        raise

if __name__ == "__main__":
    test_email_categorization()