#!/usr/bin/env python3
"""
PII anonymization module for email taxonomy pipeline.

Anonymizes sensitive information while preserving business context.
"""

import json
import re
from typing import Dict, List, Any, Set
from dataclasses import dataclass
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationRule:
    """Defines how to anonymize a specific type of PII."""
    pattern: re.Pattern
    replacement_func: callable
    confidence_threshold: str = 'medium'


class Anonymizer:
    """Anonymizes PII in email content while preserving business context."""

    def __init__(self):
        self.seen_values = {}  # Consistent replacement mapping
        self.replacement_counters = {
            'email': 0,
            'phone': 0,
            'address': 0,
            'account_number': 0,
            'tax_id': 0,
            'name': 0,
            'company': 0
        }
        self.rules = self._create_anonymization_rules()
        self.stats = {
            'emails_processed': 0,
            'emails_anonymized': 0,
            'pii_replacements': 0
        }

    def _create_anonymization_rules(self) -> Dict[str, AnonymizationRule]:
        """Create anonymization rules for each PII type."""
        return {
            'email': AnonymizationRule(
                pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                replacement_func=self._anonymize_email,
                confidence_threshold='high'
            ),
            'phone': AnonymizationRule(
                pattern=re.compile(r'(?:\+?61\s?)?(?:\(0\)|0)?[2-9]\d{8}|(?:\+?1\s?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
                replacement_func=self._anonymize_phone,
                confidence_threshold='high'
            ),
            'address': AnonymizationRule(
                pattern=re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Place|Pl|Suite|Level|Floor|Unit)\b', re.IGNORECASE),
                replacement_func=self._anonymize_address,
                confidence_threshold='medium'
            ),
            'account_number': AnonymizationRule(
                pattern=re.compile(r'\b\d{8,20}\b'),
                replacement_func=self._anonymize_account_number,
                confidence_threshold='medium'
            ),
            'tax_id': AnonymizationRule(
                pattern=re.compile(r'\b(?:ABN:?\s*)?(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b|(?:ACN:?\s*)?(\d{3}\s?\d{3}\s?\d{3})\b', re.IGNORECASE),
                replacement_func=self._anonymize_tax_id,
                confidence_threshold='medium'
            )
        }

    def _generate_consistent_replacement(self, original_value: str, prefix: str) -> str:
        """Generate consistent anonymized replacement for a value."""
        if original_value in self.seen_values:
            return self.seen_values[original_value]

        # Generate a short hash for consistency
        hash_obj = hashlib.md5(original_value.encode())
        hash_hex = hash_obj.hexdigest()[:6].upper()

        self.replacement_counters[prefix] += 1
        replacement = f"[{prefix.upper()}_{self.replacement_counters[prefix]}_{hash_hex}]"

        self.seen_values[original_value] = replacement
        return replacement

    def _anonymize_email(self, match: re.Match) -> str:
        """Anonymize email addresses while preserving domain type."""
        email = match.group(0).lower()

        # Keep Litera emails as is for direction classification
        if '@litera.com' in email:
            return email

        # Preserve common business email patterns
        if any(business in email for business in ['billing', 'support', 'noreply', 'info', 'admin']):
            business_type = next((b for b in ['billing', 'support', 'noreply', 'info', 'admin'] if b in email), 'contact')
            return f"{business_type}@[COMPANY_DOMAIN]"

        return self._generate_consistent_replacement(email, 'email')

    def _anonymize_phone(self, match: re.Match) -> str:
        """Anonymize phone numbers."""
        return self._generate_consistent_replacement(match.group(0), 'phone')

    def _anonymize_address(self, match: re.Match) -> str:
        """Anonymize street addresses."""
        return self._generate_consistent_replacement(match.group(0), 'address')

    def _anonymize_account_number(self, match: re.Match) -> str:
        """Anonymize account numbers."""
        return self._generate_consistent_replacement(match.group(0), 'account_number')

    def _anonymize_tax_id(self, match: re.Match) -> str:
        """Anonymize tax IDs (ABN/ACN)."""
        return self._generate_consistent_replacement(match.group(0), 'tax_id')

    def anonymize_text(self, text: str) -> str:
        """Anonymize PII in text content."""
        if not text:
            return text

        original_text = text

        # Apply each anonymization rule
        for rule_name, rule in self.rules.items():
            text = rule.pattern.sub(rule.replacement_func, text)

        # Track if changes were made
        if text != original_text:
            self.stats['pii_replacements'] += 1

        return text

    def anonymize_email(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize PII in a single email."""
        anonymized_email = email.copy()

        # Anonymize text fields
        text_fields = ['content', 'subject', 'message']
        for field in text_fields:
            if field in anonymized_email and anonymized_email[field]:
                anonymized_email[field] = self.anonymize_text(anonymized_email[field])

        # Anonymize sender field (but preserve @litera.com for direction classification)
        if 'sender' in anonymized_email:
            sender = anonymized_email['sender']
            if sender and '@litera.com' not in sender.lower():
                anonymized_email['sender'] = self.anonymize_text(sender)

        return anonymized_email

    def anonymize_dataset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize an entire email dataset."""
        logger.info("Starting PII anonymization...")

        anonymized_data = {
            'emails': [],
            'metadata': data.get('metadata', {}).copy(),
            'anonymization_stats': {}
        }

        # Handle both threaded and flat email structures
        emails = []
        if 'threads' in data:
            # Extract emails from threaded structure
            logger.info("Processing threaded email structure...")
            for thread in data['threads']:
                thread_emails = thread.get('emails', [])
                emails.extend(thread_emails)
            logger.info(f"Extracted {len(emails)} emails from {len(data['threads'])} threads")
        else:
            # Handle flat email structure (legacy format)
            emails = data.get('emails', [])
            logger.info(f"Processing flat email structure with {len(emails)} emails")

        self.stats['emails_processed'] = len(emails)

        for email in emails:
            original_email = json.dumps(email, sort_keys=True)
            anonymized_email = self.anonymize_email(email)
            anonymized_email_str = json.dumps(anonymized_email, sort_keys=True)

            # Track if email was changed
            if original_email != anonymized_email_str:
                self.stats['emails_anonymized'] += 1

            anonymized_data['emails'].append(anonymized_email)

        # Calculate success rate
        if self.stats['emails_processed'] > 0:
            success_rate = (self.stats['emails_anonymized'] / self.stats['emails_processed']) * 100
        else:
            success_rate = 0

        anonymized_data['anonymization_stats'] = {
            'emails_processed': self.stats['emails_processed'],
            'emails_anonymized': self.stats['emails_anonymized'],
            'pii_replacements': self.stats['pii_replacements'],
            'success_rate_percent': round(success_rate, 1),
            'replacement_counts': self.replacement_counters.copy()
        }

        logger.info(f"Anonymization complete: {self.stats['emails_anonymized']}/{self.stats['emails_processed']} emails ({success_rate:.1f}%)")

        return anonymized_data

    def save_results(self, anonymized_data: Dict[str, Any], output_path: str) -> None:
        """Save anonymized data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anonymized_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved anonymized data to {output_path}")