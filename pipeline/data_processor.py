#!/usr/bin/env python3
"""
Data processing module for email taxonomy pipeline.

Handles HTML cleaning, thread separation, and data structuring.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from html import unescape
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes raw email data for taxonomy analysis."""

    def __init__(self, clean_html: bool = True, separate_threads: bool = True):
        self.clean_html = clean_html
        self.separate_threads = separate_threads

    def clean_html_content(self, html_content: str) -> str:
        """Extract clean text content from HTML email message."""
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
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Unescape HTML entities
        text = unescape(text)

        return text

    def classify_email_direction(self, email: Dict[str, Any]) -> str:
        """Classify email as incoming or outgoing based on sender."""
        # Use @litera.com rule for direction classification
        sender = email.get('sender', email.get('from', ''))
        if '@litera.com' in sender.lower():
            return 'outgoing'
        return 'incoming'

    def separate_email_threads(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Separate threaded email conversations into individual messages."""
        separated_emails = []

        for email in emails:
            # For now, treat each email as individual
            # In a full implementation, this would parse thread structure
            separated_email = {
                'id': email.get('id'),
                'subject': email.get('subject', ''),
                'content': email.get('message', ''),
                'sender': email.get('sender', email.get('from', '')),
                'direction': self.classify_email_direction(email),
                'thread_id': email.get('thread_id', email.get('id')),
                'timestamp': email.get('timestamp', email.get('date', ''))
            }
            separated_emails.append(separated_email)

        return separated_emails

    def parse_malformed_json(self, content: str) -> List[Dict[str, Any]]:
        """Parse malformed JSON that may have unescaped quotes in string values."""
        records = []

        # Find array boundaries
        content = content.strip()
        if not (content.startswith('[') and content.endswith(']')):
            logger.error("Invalid JSON format: not an array")
            return []

        # Remove outer brackets
        inner_content = content[1:-1].strip()

        # Split by record boundaries using state machine
        record_texts = []
        depth = 0
        current_record = ""
        in_string = False
        escape_next = False

        for char in inner_content:
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

        logger.info(f"Found {len(record_texts)} potential records to parse")

        # Parse each record individually
        for i, record_text in enumerate(record_texts):
            try:
                # Wrap in braces if needed
                if not record_text.strip().startswith('{'):
                    record_text = '{' + record_text + '}'

                record = json.loads(record_text)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse record {i+1}: {e}")
                # Try to fix unescaped quotes
                try:
                    fixed_record = self._fix_unescaped_quotes(record_text)
                    record = json.loads(fixed_record)
                    records.append(record)
                    logger.info(f"Successfully repaired record {i+1}")
                except Exception as repair_error:
                    logger.error(f"Could not repair record {i+1}: {repair_error}")
                    continue

        return records

    def _fix_unescaped_quotes(self, record_text: str) -> str:
        """Attempt to fix unescaped quotes in JSON record text."""
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
                line = re.sub(r'(?<!\\)"(?!,\s*$)(?!\s*})', r'\\"', line)

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def process_emails(self, input_file: str) -> Dict[str, Any]:
        """Process raw email data through the complete pipeline."""
        logger.info(f"Processing emails from {input_file}")

        # Load raw data with robust JSON parsing
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            logger.info("Successfully parsed JSON using standard parser")
        except json.JSONDecodeError as e:
            logger.warning(f"Standard JSON parsing failed: {e}")
            logger.info("Attempting to parse malformed JSON...")

            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            raw_data = self.parse_malformed_json(content)

        # Handle different input formats
        if isinstance(raw_data, list):
            emails = raw_data
        elif isinstance(raw_data, dict) and 'emails' in raw_data:
            emails = raw_data['emails']
        else:
            raise ValueError("Input data format not recognized")

        logger.info(f"Loaded {len(emails)} raw emails")

        # Clean HTML content if requested
        if self.clean_html:
            logger.info("Cleaning HTML content...")
            for email in emails:
                if 'message' in email:
                    email['message'] = self.clean_html_content(email['message'])

        # Separate threads if requested
        if self.separate_threads:
            logger.info("Separating email threads...")
            emails = self.separate_email_threads(emails)

        # Count email directions
        incoming_count = sum(1 for email in emails if email.get('direction') == 'incoming')
        outgoing_count = len(emails) - incoming_count

        processed_data = {
            'emails': emails,
            'metadata': {
                'total_emails': len(emails),
                'incoming_emails': incoming_count,
                'outgoing_emails': outgoing_count,
                'processing_config': {
                    'clean_html': self.clean_html,
                    'separate_threads': self.separate_threads
                }
            }
        }

        logger.info(f"Processing complete: {len(emails)} emails ({incoming_count} incoming, {outgoing_count} outgoing)")

        return processed_data

    def save_results(self, processed_data: Dict[str, Any], output_path: str) -> None:
        """Save processed data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_path}")