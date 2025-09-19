#!/usr/bin/env python3
"""
Test script to verify the enhanced robust JSON parser.
"""

import json
import logging
import sys
from pathlib import Path

# Add pipeline to Python path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from data_processor import DataProcessor

def test_robust_parser():
    """Test the enhanced robust JSON parser."""

    # Setup logging to show parsing details
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Input and output paths
    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"
    output_file = "test_robust_parser_results.json"

    print("Testing enhanced robust JSON parser...")

    try:
        # Initialize data processor
        processor = DataProcessor(clean_html=True, separate_threads=True)

        # Process the emails with the enhanced parser
        processed_data = processor.process_emails(input_file)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        # Analyze parser performance
        analyze_parser_performance(processed_data)

        print(f"\n✓ Robust parser test completed!")
        print(f"Results saved to: {output_file}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_parser_performance(processed_data):
    """Analyze the performance of the enhanced parser."""

    metadata = processed_data['metadata']
    threads = processed_data['threads']

    print(f"\n=== ROBUST PARSER PERFORMANCE ===")
    print(f"Total threads: {metadata['total_threads']}")
    print(f"Relevant threads after filtering: {metadata['relevant_threads']}")
    print(f"Total emails parsed: {metadata['total_emails']}")
    print(f"Incoming emails: {metadata['incoming_emails']}")
    print(f"Outgoing emails: {metadata['outgoing_emails']}")

    # Calculate data recovery improvement
    # Previous parser typically got ~4,697 emails with ~303 failures
    # Let's see if we improved on that
    total_processed = metadata['total_emails']

    print(f"\n=== DATA RECOVERY ANALYSIS ===")
    print(f"Total emails successfully parsed: {total_processed}")

    if total_processed > 4697:
        recovered_emails = total_processed - 4697
        print(f"🎉 IMPROVEMENT: Recovered {recovered_emails} additional emails!")
        print(f"📈 Recovery improvement: {recovered_emails/4697*100:.1f}%")
    elif total_processed == 4697:
        print("📊 Same count as before - no additional data loss/recovery")
    else:
        print(f"⚠️  Fewer emails than before: {4697 - total_processed} difference")

    # Analyze thread quality
    print(f"\n=== THREAD QUALITY ===")
    multi_email_threads = [t for t in threads if len(t['emails']) > 1]
    single_email_threads = [t for t in threads if len(t['emails']) == 1]

    print(f"Multi-email threads: {len(multi_email_threads)}")
    print(f"Single-email threads: {len(single_email_threads)}")

    # Look for threads with good conversation patterns
    good_conversations = 0
    for thread in multi_email_threads:
        directions = [email['direction'] for email in thread['emails']]
        # Count direction changes (good sign of conversation)
        changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
        if changes > 0:
            good_conversations += 1

    if multi_email_threads:
        conversation_quality = good_conversations / len(multi_email_threads) * 100
        print(f"Conversation quality: {conversation_quality:.1f}% have direction changes")

    print(f"\n=== SAMPLE SUCCESSFUL THREADS ===")
    # Show some examples of successfully parsed threads
    for i, thread in enumerate(threads[:3]):
        emails_count = len(thread['emails'])
        subject = thread.get('subject', 'No subject')[:50]
        directions = [email['direction'] for email in thread['emails']]
        pattern = ' -> '.join(directions)

        print(f"{i+1}. Thread {thread['thread_id']}: {emails_count} emails")
        print(f"   Subject: {subject}...")
        print(f"   Pattern: {pattern}")

if __name__ == "__main__":
    success = test_robust_parser()
    if success:
        print("\n✓ Robust parser test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Robust parser test failed!")
        sys.exit(1)