#!/usr/bin/env python3
"""
Test script to verify the enhanced email classification accuracy.
"""

import json
import logging
import sys
from pathlib import Path

# Add pipeline to Python path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from data_processor import DataProcessor

def test_enhanced_classification():
    """Test the enhanced classification system."""

    # Setup logging to show classification info
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Input and output paths
    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"
    output_file = "test_enhanced_classification_results.json"

    print("Testing enhanced email classification system...")

    try:
        # Initialize data processor with enhanced classification
        processor = DataProcessor(clean_html=True, separate_threads=True)

        # Process the emails
        processed_data = processor.process_emails(input_file)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        # Analyze classification results
        analyze_classification_quality(processed_data)

        print(f"\n✓ Enhanced classification test completed!")
        print(f"Results saved to: {output_file}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_classification_quality(processed_data):
    """Analyze the quality of email classifications."""

    metadata = processed_data['metadata']
    threads = processed_data['threads']

    print(f"\n=== ENHANCED CLASSIFICATION RESULTS ===")
    print(f"Total threads before filtering: {metadata['total_threads']}")
    print(f"Relevant threads after filtering: {metadata['relevant_threads']}")
    print(f"Filtered out (single outgoing): {metadata['filtered_threads']}")
    print(f"Total emails: {metadata['total_emails']}")
    print(f"Incoming emails: {metadata['incoming_emails']}")
    print(f"Outgoing emails: {metadata['outgoing_emails']}")

    # Analyze thread patterns
    print(f"\n=== THREAD PATTERN ANALYSIS ===")

    thread_patterns = {
        'pure_conversations': 0,      # Alternating incoming/outgoing
        'invoice_with_responses': 0,  # Outgoing first, then incoming replies
        'all_outgoing': 0,           # All outgoing (suspicious if multi-email)
        'all_incoming': 0,           # All incoming (also suspicious)
        'mixed_patterns': 0          # Other patterns
    }

    conversation_quality_scores = []

    for thread in threads:
        emails = thread['emails']
        if len(emails) == 1:
            continue

        directions = [email['direction'] for email in emails]

        # Classify thread pattern
        if all(d == 'outgoing' for d in directions):
            thread_patterns['all_outgoing'] += 1
        elif all(d == 'incoming' for d in directions):
            thread_patterns['all_incoming'] += 1
        elif directions[0] == 'outgoing' and all(d == 'incoming' for d in directions[1:]):
            thread_patterns['invoice_with_responses'] += 1
        else:
            # Check for alternating pattern (genuine conversation)
            alternating_score = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
            if alternating_score >= len(directions) * 0.4:
                thread_patterns['pure_conversations'] += 1
                conversation_quality_scores.append(alternating_score / len(directions))
            else:
                thread_patterns['mixed_patterns'] += 1

    total_multi_email_threads = sum(thread_patterns.values())

    print(f"Multi-email threads analyzed: {total_multi_email_threads}")
    print(f"Pure conversations (alternating): {thread_patterns['pure_conversations']} ({thread_patterns['pure_conversations']/total_multi_email_threads*100:.1f}%)")
    print(f"Invoice with responses: {thread_patterns['invoice_with_responses']} ({thread_patterns['invoice_with_responses']/total_multi_email_threads*100:.1f}%)")
    print(f"All outgoing (suspicious): {thread_patterns['all_outgoing']} ({thread_patterns['all_outgoing']/total_multi_email_threads*100:.1f}%)")
    print(f"All incoming (suspicious): {thread_patterns['all_incoming']} ({thread_patterns['all_incoming']/total_multi_email_threads*100:.1f}%)")
    print(f"Mixed patterns: {thread_patterns['mixed_patterns']} ({thread_patterns['mixed_patterns']/total_multi_email_threads*100:.1f}%)")

    # Quality metrics
    suspicious_threads = thread_patterns['all_outgoing'] + thread_patterns['all_incoming']
    quality_score = (total_multi_email_threads - suspicious_threads) / total_multi_email_threads * 100

    print(f"\n=== QUALITY METRICS ===")
    print(f"Classification quality score: {quality_score:.1f}%")
    print(f"Suspicious patterns (all same direction): {suspicious_threads}/{total_multi_email_threads}")

    if conversation_quality_scores:
        avg_conversation_quality = sum(conversation_quality_scores) / len(conversation_quality_scores) * 100
        print(f"Average conversation alternation quality: {avg_conversation_quality:.1f}%")

    # Show examples of different thread types
    print(f"\n=== SAMPLE THREADS BY TYPE ===")

    examples_shown = {'pure_conversations': 0, 'invoice_with_responses': 0, 'all_outgoing': 0}

    for thread in threads[:20]:  # Look at first 20 threads
        emails = thread['emails']
        if len(emails) <= 1:
            continue

        directions = [email['direction'] for email in emails]
        thread_type = None

        if all(d == 'outgoing' for d in directions):
            thread_type = 'all_outgoing'
        elif directions[0] == 'outgoing' and all(d == 'incoming' for d in directions[1:]):
            thread_type = 'invoice_with_responses'
        else:
            alternating_score = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
            if alternating_score >= len(directions) * 0.4:
                thread_type = 'pure_conversations'

        if thread_type and examples_shown[thread_type] < 2:
            examples_shown[thread_type] += 1
            subject = thread.get('subject', 'No subject')[:60]
            print(f"\n{thread_type.upper()}: Thread {thread['thread_id']}")
            print(f"  Subject: {subject}...")
            print(f"  Pattern: {' -> '.join(directions)}")

            # Show first email content snippet for context
            if emails:
                content_snippet = emails[0].get('content', '')[:100].replace('\n', ' ')
                print(f"  First email: {content_snippet}...")

if __name__ == "__main__":
    success = test_enhanced_classification()
    if success:
        print("\n✓ Enhanced classification test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Enhanced classification test failed!")
        sys.exit(1)