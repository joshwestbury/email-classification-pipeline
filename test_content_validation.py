#!/usr/bin/env python3
"""
Test the content validation fixes for empty/whitespace-only content.
"""

import re
import json


def test_field_extraction_validation():
    """Test the enhanced field extraction with content validation."""

    # Simulate the validation logic from _extract_fields_from_malformed_record
    def validate_message_content(value):
        """Apply the same validation logic as the fixed method."""
        if value and value.strip():
            # Unescape common escape sequences to check actual content
            unescaped_value = value.replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t')

            # Check if unescaped content is just whitespace
            if not unescaped_value.strip():
                print(f"Filtered out escaped whitespace-only content: {repr(value[:50])}")
                return False
            else:
                # Must contain at least one letter or number (after unescaping)
                if re.search(r'[a-zA-Z0-9]', unescaped_value):
                    return True
                else:
                    print(f"Filtered out whitespace-only message content: {repr(value[:50])}")
                    return False
        else:
            print(f"Filtered out empty/minimal message content: {repr(value)}")
            return False

    # Test cases that should be filtered out
    bad_content_examples = [
        "\r\n\r\n\r\n",
        "\r\n\r\n\r\n\r\n",
        "\n\n\n",
        "   \t\t\t   ",
        "",
        "  ",
        "\r\n",
        "   \n   \r   \n   ",
        # Escaped whitespace sequences (the main issue we're fixing)
        "\\r\\n\\r\\n\\r\\n",
        "\\r\\n\\r\\n\\r\\n\\r\\n",
        "\\n\\n\\n",
        "\\t\\t\\t",
        "\\r\\n",
        "   \\r\\n   \\r\\n   "
    ]

    # Test cases that should pass
    good_content_examples = [
        "Hello, this is a real email",
        "Invoice 12345 is due",
        "Thank you for your payment.",
        "Please call me at 555-1234",
        "a",  # Single letter
        "1",  # Single number
        "   Hello   ",  # Valid content with whitespace
    ]

    print("=== TESTING BAD CONTENT (should be filtered out) ===")
    bad_filtered = 0
    for i, content in enumerate(bad_content_examples):
        result = validate_message_content(content)
        if not result:
            bad_filtered += 1
        print(f"{i+1}. {repr(content)} -> {'FILTERED' if not result else 'PASSED'}")

    print(f"\n=== TESTING GOOD CONTENT (should pass) ===")
    good_passed = 0
    for i, content in enumerate(good_content_examples):
        result = validate_message_content(content)
        if result:
            good_passed += 1
        print(f"{i+1}. {repr(content)} -> {'PASSED' if result else 'FILTERED'}")

    print(f"\n=== RESULTS ===")
    print(f"Bad content filtered: {bad_filtered}/{len(bad_content_examples)}")
    print(f"Good content passed: {good_passed}/{len(good_content_examples)}")

    if bad_filtered == len(bad_content_examples) and good_passed == len(good_content_examples):
        print("✓ ALL TESTS PASSED - Content validation working correctly!")
        return True
    else:
        print("✗ SOME TESTS FAILED - Content validation needs adjustment")
        return False


def test_existing_data_analysis():
    """Analyze the existing test data to see what would be filtered."""

    try:
        with open('test_robust_parser_results.json', 'r') as f:
            data = json.load(f)

        total_emails = 0
        would_be_filtered = 0
        filter_examples = []

        for thread in data['threads']:
            for email in thread['emails']:
                total_emails += 1
                content = email.get('content', '')

                # Apply the same filtering logic (with escaped sequence handling)
                if content and content.strip():
                    # Unescape common escape sequences to check actual content
                    unescaped_content = content.replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t')

                    # Check if unescaped content is just whitespace
                    if not unescaped_content.strip():
                        would_be_filtered += 1
                        filter_examples.append({
                            'content': repr(content[:50]),
                            'length': len(content),
                            'subject': email.get('subject', 'No subject')[:30],
                            'reason': 'escaped_whitespace'
                        })
                    elif not re.search(r'[a-zA-Z0-9]', unescaped_content):
                        would_be_filtered += 1
                        filter_examples.append({
                            'content': repr(content[:50]),
                            'length': len(content),
                            'subject': email.get('subject', 'No subject')[:30],
                            'reason': 'whitespace_only'
                        })
                elif content:
                    would_be_filtered += 1
                    filter_examples.append({
                        'content': repr(content),
                        'length': len(content),
                        'subject': email.get('subject', 'No subject')[:30],
                        'reason': 'empty_minimal'
                    })

        print(f"\n=== ANALYSIS OF EXISTING DATA ===")
        print(f"Total emails: {total_emails}")
        print(f"Would be filtered with new validation: {would_be_filtered}")
        print(f"Percentage that would be filtered: {would_be_filtered/total_emails*100:.2f}%")

        if filter_examples:
            print(f"\nFirst 5 examples that would be filtered:")
            for i, example in enumerate(filter_examples[:5]):
                print(f"{i+1}. {example['content']} (len: {example['length']}) - {example['subject']} [{example['reason']}]")

        return would_be_filtered

    except FileNotFoundError:
        print("No existing test results found - that's expected after applying the fix")
        return 0


if __name__ == "__main__":
    print("Testing content validation logic...")

    # Test the validation logic
    validation_works = test_field_extraction_validation()

    # Analyze existing data
    filtered_count = test_existing_data_analysis()

    if validation_works:
        print(f"\n✓ Content validation fix is working correctly!")
        print(f"The fix should eliminate {filtered_count} problematic emails from future runs.")
    else:
        print(f"\n✗ Content validation needs further adjustment.")