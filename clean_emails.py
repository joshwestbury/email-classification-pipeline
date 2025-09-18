#!/usr/bin/env python3
"""
Script to clean up collection email JSON data.

This script processes the CollectionNotes_SentimentAnalysis_SampleEmails.json file
to extract clean email content by:
1. Maintaining the 'id' and 'subject' properties
2. Stripping HTML tags from the 'message' property
3. Extracting only the actual email content text
4. Saving the cleaned data to a new JSON file
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from html import unescape
from bs4 import BeautifulSoup


def clean_html_content(html_content: str) -> str:
    """
    Extract clean text content from HTML email message.

    Args:
        html_content: Raw HTML content from email message

    Returns:
        Clean text content with HTML tags removed and whitespace normalized
    """
    if not html_content:
        return ""

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text content
    text = soup.get_text()

    # Clean up whitespace
    # Replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Unescape HTML entities
    text = unescape(text)

    return text


def clean_email_data(email_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a single email record.

    Args:
        email_record: Dictionary containing id, subject, and message

    Returns:
        Cleaned email record with HTML stripped from message
    """
    cleaned_record = {
        'id': email_record.get('id'),
        'subject': email_record.get('subject', ''),
        'message': clean_html_content(email_record.get('message', ''))
    }

    return cleaned_record


def parse_malformed_json(content: str) -> List[Dict[str, Any]]:
    """
    Parse malformed JSON that may have unescaped quotes in string values.

    Args:
        content: Raw JSON content as string

    Returns:
        List of parsed email records
    """
    import ast

    # Try a more aggressive approach: use ast.literal_eval on individual records
    # First, let's try to split into individual records
    records = []

    # Find array boundaries
    content = content.strip()
    if not (content.startswith('[') and content.endswith(']')):
        print("Invalid JSON format: not an array")
        return []

    # Remove outer brackets
    inner_content = content[1:-1].strip()

    # Split by record boundaries (looking for },\s*{)
    import re

    # More robust splitting approach
    record_texts = []
    depth = 0
    current_record = ""
    in_string = False
    escape_next = False

    for i, char in enumerate(inner_content):
        if escape_next:
            current_record += char
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            current_record += char
            continue

        if char == '"' and not escape_next:
            in_string = not in_string

        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1

        current_record += char

        # If we're at depth 0 and find a comma, we've completed a record
        if depth == 0 and char == ',' and not in_string:
            record_texts.append(current_record[:-1].strip())  # Remove the comma
            current_record = ""

    # Add the last record
    if current_record.strip():
        record_texts.append(current_record.strip())

    print(f"Found {len(record_texts)} potential records")

    # Now try to parse each record individually
    for i, record_text in enumerate(record_texts):
        try:
            # Wrap in braces if needed
            if not record_text.strip().startswith('{'):
                record_text = '{' + record_text + '}'

            record = json.loads(record_text)
            records.append(record)
        except json.JSONDecodeError as e:
            print(f"Failed to parse record {i+1}: {e}")
            # Try a more aggressive fix for unescaped quotes
            try:
                # Simple approach: replace unescaped quotes in HTML content
                fixed_record = fix_unescaped_quotes(record_text)
                record = json.loads(fixed_record)
                records.append(record)
                print(f"Successfully repaired record {i+1}")
            except Exception as repair_error:
                print(f"Could not repair record {i+1}: {repair_error}")
                continue

    return records


def fix_unescaped_quotes(record_text: str) -> str:
    """
    Attempt to fix unescaped quotes in JSON record text.

    Args:
        record_text: JSON record as string

    Returns:
        Fixed JSON string
    """
    # This is a heuristic approach - look for the message field and escape quotes within it
    lines = record_text.split('\n')
    fixed_lines = []
    in_message_field = False

    for line in lines:
        if '"message":' in line:
            in_message_field = True
        elif in_message_field and line.strip().startswith('"') and (line.strip().endswith('"}') or line.strip().endswith('",')):
            in_message_field = False

        if in_message_field and '"message":' not in line:
            # Inside message field - escape unescaped quotes
            # This is a simple heuristic - escape quotes that aren't already escaped
            line = re.sub(r'(?<!\\)"(?!,\s*$)(?!\s*})', r'\\"', line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def main():
    """Main function to process the email JSON file."""
    # Define file paths
    input_file = Path("Collection Notes - Sentiment Analysis/CollectionNotes_SentimentAnalysis_SampleEmails.json")
    output_file = Path("emails_cleaned.json")

    try:
        # Read the input JSON file
        print(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to parse JSON with error handling and repair if needed
        try:
            email_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error detected. Attempting to repair...")
            print(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")

            # Try to read the file using a more lenient approach
            # Since the JSON seems to have unescaped quotes in HTML content,
            # we'll use a different strategy
            email_data = parse_malformed_json(content)
            if not email_data:
                raise

        print(f"Loaded {len(email_data)} email records")

        # Clean each email record
        cleaned_emails = []
        for i, email_record in enumerate(email_data):
            if i % 100 == 0:  # Progress indicator
                print(f"Processing email {i+1}/{len(email_data)}")

            cleaned_record = clean_email_data(email_record)
            cleaned_emails.append(cleaned_record)

        # Write cleaned data to output file
        print(f"Writing cleaned data to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_emails, f, indent=2, ensure_ascii=False)

        print(f"Successfully cleaned {len(cleaned_emails)} email records")
        print(f"Output saved to: {output_file}")

        # Show sample of cleaned data
        if cleaned_emails:
            print("\nSample of cleaned data:")
            sample = cleaned_emails[0]
            print(f"ID: {sample['id']}")
            print(f"Subject: {sample['subject']}")
            print(f"Message preview: {sample['message'][:200]}...")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()