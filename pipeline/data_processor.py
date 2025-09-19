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

    def process_emails(self, input_file: str) -> Dict[str, Any]:
        """Process raw email data through the complete pipeline."""
        logger.info(f"Processing emails from {input_file}")

        # Load raw data
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

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