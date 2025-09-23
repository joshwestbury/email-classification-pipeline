#!/usr/bin/env python3
"""
PII anonymization module for email taxonomy pipeline.

Anonymizes sensitive information while preserving business context.
Implements secure salted SHA256 hashing for consistent anonymization.
"""

import json
import re
import os
import secrets
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for PII detection."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AnonymizationRule:
    """Defines how to anonymize a specific type of PII."""
    pattern: re.Pattern
    replacement_func: callable
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    description: str = ""


@dataclass
class PIIDetection:
    """Represents a detected PII instance."""
    text: str
    type: str
    confidence: ConfidenceLevel
    position: Tuple[int, int]
    replacement: str = ""


class Anonymizer:
    """Anonymizes PII in email content while preserving business context.

    Uses salted SHA256 hashing for secure, consistent anonymization.
    Implements confidence-based tiering for PII detection.
    """

    SALT_FILE = ".anonymization_salt"  # Hidden file for persistent salt
    SALT_LENGTH = 32  # 256 bits of randomness

    def __init__(self, salt_path: Optional[str] = None, confidence_threshold: ConfidenceLevel = ConfidenceLevel.MEDIUM):
        """Initialize anonymizer with persistent salt management.

        Args:
            salt_path: Path to salt file. If None, uses default location.
            confidence_threshold: Minimum confidence level for PII detection.
        """
        self.salt = self._load_or_create_salt(salt_path)
        self.confidence_threshold = confidence_threshold
        self.seen_values = {}  # Consistent replacement mapping
        self.replacement_counters = {
            'email': 0,
            'phone': 0,
            'address': 0,
            'postal_code': 0,
            'credit_card': 0,
            'iban': 0,
            'account_number': 0,
            'tax_id': 0,
            'ssn': 0,
            'name': 0,
            'company': 0
        }
        self.rules = self._create_anonymization_rules()
        self.stats = {
            'emails_processed': 0,
            'emails_anonymized': 0,
            'pii_replacements': 0,
            'pii_by_type': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }

    def _load_or_create_salt(self, salt_path: Optional[str] = None) -> bytes:
        """Load existing salt or create a new one.

        Args:
            salt_path: Path to salt file.

        Returns:
            Salt bytes for consistent hashing.
        """
        if salt_path is None:
            salt_path = self.SALT_FILE

        salt_file = Path(salt_path)

        # Try to load existing salt
        if salt_file.exists():
            try:
                with open(salt_file, 'rb') as f:
                    salt = f.read()
                    if len(salt) == self.SALT_LENGTH:
                        logger.info(f"Loaded existing salt from {salt_file}")
                        return salt
                    else:
                        logger.warning(f"Invalid salt length in {salt_file}, creating new salt")
            except Exception as e:
                logger.error(f"Error loading salt: {e}")

        # Create new salt
        salt = secrets.token_bytes(self.SALT_LENGTH)

        # Save salt for future use
        try:
            with open(salt_file, 'wb') as f:
                f.write(salt)
            # Set restrictive permissions (owner read/write only)
            os.chmod(salt_file, 0o600)
            logger.info(f"Created new salt and saved to {salt_file}")
        except Exception as e:
            logger.warning(f"Could not save salt to file: {e}. Using temporary salt.")

        return salt

    def _create_anonymization_rules(self) -> Dict[str, AnonymizationRule]:
        """Create comprehensive anonymization rules for each PII type."""
        return {
            'email': AnonymizationRule(
                pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                replacement_func=self._anonymize_email,
                confidence_level=ConfidenceLevel.HIGH,
                description="Email addresses"
            ),
            'phone': AnonymizationRule(
                pattern=re.compile(r'(?:\+?61\s?)?(?:\(0\)|0)?[2-9]\d{8}|(?:\+?1\s?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|(?:\+44\s?)?(?:\(0\)|0)?[1-9]\d{9,10}'),
                replacement_func=self._anonymize_phone,
                confidence_level=ConfidenceLevel.HIGH,
                description="Phone numbers (US, AU, UK)"
            ),
            'credit_card': AnonymizationRule(
                pattern=re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b'),
                replacement_func=self._anonymize_credit_card,
                confidence_level=ConfidenceLevel.HIGH,
                description="Credit card numbers"
            ),
            'iban': AnonymizationRule(
                pattern=re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b'),
                replacement_func=self._anonymize_iban,
                confidence_level=ConfidenceLevel.HIGH,
                description="IBAN account numbers"
            ),
            'ssn': AnonymizationRule(
                pattern=re.compile(r'\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b'),
                replacement_func=self._anonymize_ssn,
                confidence_level=ConfidenceLevel.MEDIUM,
                description="US Social Security Numbers"
            ),
            'postal_code': AnonymizationRule(
                pattern=re.compile(r'\b(?:[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}|\d{5}(?:-\d{4})?|[A-Z]\d[A-Z]\s?\d[A-Z]\d|\d{4})\b', re.IGNORECASE),
                replacement_func=self._anonymize_postal_code,
                confidence_level=ConfidenceLevel.LOW,
                description="Postal/ZIP codes (US, UK, CA, AU)"
            ),
            'address': AnonymizationRule(
                pattern=re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Place|Pl|Court|Ct|Way|Circle|Cir|Square|Sq|Parkway|Pkwy|Terrace|Ter|Suite|Level|Floor|Unit|Apt|Apartment)\b', re.IGNORECASE),
                replacement_func=self._anonymize_address,
                confidence_level=ConfidenceLevel.MEDIUM,
                description="Street addresses"
            ),
            'account_number': AnonymizationRule(
                pattern=re.compile(r'\b(?:Account|Acct|A/C)[:\s#]*\d{6,20}\b|\b\d{8,20}\b', re.IGNORECASE),
                replacement_func=self._anonymize_account_number,
                confidence_level=ConfidenceLevel.MEDIUM,
                description="Account numbers"
            ),
            'tax_id': AnonymizationRule(
                pattern=re.compile(r'\b(?:ABN:?\s*)?(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b|(?:ACN:?\s*)?(\d{3}\s?\d{3}\s?\d{3})\b|(?:EIN:?\s*)?(\d{2}-\d{7})\b', re.IGNORECASE),
                replacement_func=self._anonymize_tax_id,
                confidence_level=ConfidenceLevel.MEDIUM,
                description="Tax IDs (ABN, ACN, EIN)"
            )
        }

    def _generate_consistent_replacement(self, original_value: str, prefix: str) -> str:
        """Generate consistent anonymized replacement using salted SHA256.

        Args:
            original_value: The PII value to anonymize.
            prefix: Type prefix for the replacement.

        Returns:
            Consistent anonymized replacement string.
        """
        if original_value in self.seen_values:
            return self.seen_values[original_value]

        # Use salted SHA256 for secure hashing
        salted_value = self.salt + original_value.encode('utf-8')
        hash_obj = hashlib.sha256(salted_value)
        hash_hex = hash_obj.hexdigest()[:8].upper()  # Use 8 chars for better uniqueness

        self.replacement_counters[prefix] += 1
        replacement = f"[{prefix.upper()}_{self.replacement_counters[prefix]}_{hash_hex}]"

        self.seen_values[original_value] = replacement

        # Update stats
        if prefix not in self.stats['pii_by_type']:
            self.stats['pii_by_type'][prefix] = 0
        self.stats['pii_by_type'][prefix] += 1

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
        """Anonymize tax IDs (ABN/ACN/EIN)."""
        return self._generate_consistent_replacement(match.group(0), 'tax_id')

    def _anonymize_credit_card(self, match: re.Match) -> str:
        """Anonymize credit card numbers."""
        return self._generate_consistent_replacement(match.group(0), 'credit_card')

    def _anonymize_iban(self, match: re.Match) -> str:
        """Anonymize IBAN numbers."""
        return self._generate_consistent_replacement(match.group(0), 'iban')

    def _anonymize_ssn(self, match: re.Match) -> str:
        """Anonymize Social Security Numbers."""
        return self._generate_consistent_replacement(match.group(0), 'ssn')

    def _anonymize_postal_code(self, match: re.Match) -> str:
        """Anonymize postal/ZIP codes."""
        return self._generate_consistent_replacement(match.group(0), 'postal_code')

    def detect_pii(self, text: str) -> List[PIIDetection]:
        """Detect all PII in text with confidence levels.

        Args:
            text: Text to analyze.

        Returns:
            List of PII detections with confidence levels.
        """
        detections = []

        for rule_name, rule in self.rules.items():
            for match in rule.pattern.finditer(text):
                detection = PIIDetection(
                    text=match.group(0),
                    type=rule_name,
                    confidence=rule.confidence_level,
                    position=(match.start(), match.end())
                )
                detections.append(detection)

        return detections

    def anonymize_text(self, text: str, apply_confidence_filter: bool = True) -> str:
        """Anonymize PII in text content with confidence-based filtering.

        Args:
            text: Text to anonymize.
            apply_confidence_filter: Whether to filter by confidence threshold.

        Returns:
            Anonymized text.
        """
        if not text:
            return text

        original_text = text

        # Process rules in order of confidence (high to low)
        for rule_name, rule in sorted(self.rules.items(),
                                     key=lambda x: x[1].confidence_level.value,
                                     reverse=True):
            # Skip if below confidence threshold
            if apply_confidence_filter:
                if self._confidence_below_threshold(rule.confidence_level):
                    continue

            # Apply anonymization
            text = rule.pattern.sub(rule.replacement_func, text)

            # Track confidence distribution
            if text != original_text:
                self.stats['confidence_distribution'][rule.confidence_level.value] += 1

        # Track if changes were made
        if text != original_text:
            self.stats['pii_replacements'] += 1

        return text

    def _confidence_below_threshold(self, confidence: ConfidenceLevel) -> bool:
        """Check if confidence level is below threshold.

        Args:
            confidence: Confidence level to check.

        Returns:
            True if below threshold.
        """
        confidence_order = {ConfidenceLevel.LOW: 1, ConfidenceLevel.MEDIUM: 2, ConfidenceLevel.HIGH: 3}
        return confidence_order[confidence] < confidence_order[self.confidence_threshold]

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
            'replacement_counts': self.replacement_counters.copy(),
            'pii_by_type': self.stats['pii_by_type'].copy(),
            'confidence_distribution': self.stats['confidence_distribution'].copy(),
            'confidence_threshold': self.confidence_threshold.value,
            'salt_configured': bool(self.salt)
        }

        logger.info(f"Anonymization complete: {self.stats['emails_anonymized']}/{self.stats['emails_processed']} emails ({success_rate:.1f}%)")

        return anonymized_data

    def save_results(self, anonymized_data: Dict[str, Any], output_path: str) -> None:
        """Save anonymized data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anonymized_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved anonymized data to {output_path}")