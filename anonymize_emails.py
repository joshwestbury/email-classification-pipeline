#!/usr/bin/env python3
"""
Email Anonymization Script

Anonymizes sensitive PII in collection emails while preserving
the structure and intent for taxonomy development.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass
import hashlib


@dataclass
class AnonymizationRule:
    """Defines how to anonymize a specific type of PII."""
    pattern: re.Pattern
    replacement_func: callable
    confidence_threshold: str = 'medium'  # minimum confidence to anonymize


class EmailAnonymizer:
    """Anonymizes PII in email content while preserving business context."""

    def __init__(self):
        self.seen_values = {}  # Consistent replacement mapping
        self.replacement_counters = {
            'email': 0,
            'phone': 0,
            'address': 0,
            'account_number': 0,
            'tax_id': 0
        }
        self.rules = self._create_anonymization_rules()

    def _create_anonymization_rules(self) -> Dict[str, AnonymizationRule]:
        """Create anonymization rules for each PII type."""
        return {
            'email': AnonymizationRule(
                pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                replacement_func=self._anonymize_email,
                confidence_threshold='high'
            ),
            'phone': AnonymizationRule(
                pattern=re.compile(r'\+?(?:61|1)?[-.\s]?\(?[0-9]{1,4}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'),
                replacement_func=self._anonymize_phone,
                confidence_threshold='high'
            ),
            'address': AnonymizationRule(
                pattern=re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Place|Pl)\b', re.IGNORECASE),
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
        email = match.group(0)

        # Preserve common business email patterns
        if any(business in email.lower() for business in ['billing', 'support', 'noreply', 'info', 'admin']):
            domain = email.split('@')[1] if '@' in email else 'company.com'
            business_type = next((b for b in ['billing', 'support', 'noreply', 'info', 'admin'] if b in email.lower()), 'contact')
            return f"{business_type}@[COMPANY_DOMAIN]"

        return self._generate_consistent_replacement(email, 'email')

    def _anonymize_phone(self, match: re.Match) -> str:
        """Anonymize phone numbers while preserving format."""
        phone = match.group(0)
        return self._generate_consistent_replacement(phone, 'phone')

    def _anonymize_address(self, match: re.Match) -> str:
        """Anonymize street addresses while preserving structure."""
        address = match.group(0)
        # Extract street type
        street_types = ['Street', 'St', 'Avenue', 'Ave', 'Road', 'Rd', 'Drive', 'Dr', 'Lane', 'Ln', 'Boulevard', 'Blvd', 'Place', 'Pl']
        street_type = next((st for st in street_types if st.lower() in address.lower()), 'Street')

        replacement = self._generate_consistent_replacement(address, 'address')
        return f"[ADDRESS_{self.replacement_counters['address']} {street_type}]"

    def _anonymize_account_number(self, match: re.Match) -> str:
        """Anonymize account numbers."""
        account = match.group(0)
        return self._generate_consistent_replacement(account, 'account_number')

    def _anonymize_tax_id(self, match: re.Match) -> str:
        """Anonymize tax/business IDs."""
        tax_id = match.group(1) or match.group(2) or match.group(0)

        # Determine type based on pattern
        if len(tax_id.replace(' ', '')) == 11:
            id_type = 'ABN'
        elif len(tax_id.replace(' ', '')) == 9:
            id_type = 'ACN'
        else:
            id_type = 'TAX_ID'

        return f"[{id_type}_{self.replacement_counters['tax_id'] + 1}]"

    def _should_anonymize_pii(self, pii_match: Dict[str, Any]) -> bool:
        """Determine if a PII match should be anonymized based on confidence and type."""
        pii_type = pii_match['type']
        confidence = pii_match['confidence']

        # Skip low-confidence matches for certain types to avoid false positives
        if pii_type in ['account_number', 'tax_id'] and confidence == 'low':
            return False

        # Skip invoice references (legitimate business data)
        if pii_type == 'invoice_reference':
            return False

        # Skip SSN (mostly false positives in this dataset)
        if pii_type == 'ssn':
            return False

        # Skip name patterns (too many false positives)
        if pii_type == 'name_patterns':
            return False

        return True

    def anonymize_text_with_pii_data(self, text: str, pii_matches: List[Dict[str, Any]]) -> str:
        """Anonymize text using pre-detected PII matches."""
        if not text:
            return text

        # Sort matches by position (descending) to avoid offset issues
        sorted_matches = sorted(
            [m for m in pii_matches if self._should_anonymize_pii(m)],
            key=lambda x: x.get('start_pos', 0),
            reverse=True
        )

        anonymized_text = text

        for pii_match in sorted_matches:
            pii_type = pii_match['type']
            original_value = pii_match['value']

            if pii_type in self.rules:
                rule = self.rules[pii_type]

                # Find all occurrences of this value in the text
                pattern = re.escape(original_value)
                matches = list(re.finditer(pattern, anonymized_text))

                # Replace from right to left to maintain positions
                for match in reversed(matches):
                    replacement = rule.replacement_func(match)
                    start, end = match.span()
                    anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]

        return anonymized_text

    def anonymize_text_patterns(self, text: str) -> str:
        """Additional pattern-based anonymization for missed PII."""
        if not text:
            return text

        anonymized = text

        # Anonymize remaining postal codes
        anonymized = re.sub(r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b', '[POSTAL_CODE]', anonymized)
        anonymized = re.sub(r'\b\d{4,5}(?:-\d{4})?\b', '[POSTAL_CODE]', anonymized)

        # Anonymize remaining long number sequences that might be accounts
        anonymized = re.sub(r'\b\d{10,15}\b', '[REFERENCE_NUMBER]', anonymized)

        return anonymized


def load_pii_data(pii_file: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Load PII detection results indexed by email ID."""
    with open(pii_file, 'r', encoding='utf-8') as f:
        pii_data = json.load(f)

    # Index PII matches by email ID
    pii_by_email = {}
    for email in pii_data['emails_with_pii']:
        email_id = email['id']
        pii_by_email[email_id] = email['pii_detected']

    return pii_by_email


def anonymize_emails(emails_file: Path, pii_file: Path, output_file: Path):
    """Anonymize emails using PII detection results."""
    print(f"Loading emails from {emails_file}")
    with open(emails_file, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    print(f"Loading PII data from {pii_file}")
    pii_by_email = load_pii_data(pii_file)

    anonymizer = EmailAnonymizer()
    anonymized_emails = []

    print(f"Anonymizing {len(emails)} emails...")

    for i, email in enumerate(emails):
        if i % 500 == 0:
            print(f"Processed {i}/{len(emails)} emails...")

        email_id = email.get('id')
        subject = email.get('subject', '')
        message = email.get('message', '')

        # Get PII matches for this email
        pii_matches = pii_by_email.get(email_id, [])

        # Anonymize subject and message using PII data
        anonymized_subject = anonymizer.anonymize_text_with_pii_data(subject, pii_matches)
        anonymized_message = anonymizer.anonymize_text_with_pii_data(message, pii_matches)

        # Apply additional pattern-based anonymization
        anonymized_subject = anonymizer.anonymize_text_patterns(anonymized_subject)
        anonymized_message = anonymizer.anonymize_text_patterns(anonymized_message)

        anonymized_emails.append({
            'id': email_id,
            'subject': anonymized_subject,
            'message': anonymized_message,
            'original_subject': subject,  # Keep for verification
            'original_message': message,  # Keep for verification
            'pii_anonymized': len(pii_matches) > 0
        })

    print(f"Saving anonymized emails to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(anonymized_emails, f, indent=2, ensure_ascii=False)

    # Generate anonymization report
    anonymized_count = sum(1 for e in anonymized_emails if e['pii_anonymized'])

    print(f"\n{'='*50}")
    print("ANONYMIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total emails processed: {len(emails)}")
    print(f"Emails with anonymized content: {anonymized_count}")
    print(f"Anonymization rate: {anonymized_count/len(emails)*100:.1f}%")
    print(f"\nReplacement counts:")
    for pii_type, count in anonymizer.replacement_counters.items():
        if count > 0:
            print(f"  {pii_type}: {count}")

    print(f"\nAnonymized dataset saved to: {output_file}")


def main():
    """Main function to anonymize email dataset."""
    emails_file = Path("emails_cleaned.json")
    pii_file = Path("emails_with_pii.json")
    output_file = Path("emails_anonymized.json")

    if not emails_file.exists():
        print(f"Error: {emails_file} not found")
        return

    if not pii_file.exists():
        print(f"Error: {pii_file} not found")
        return

    try:
        anonymize_emails(emails_file, pii_file, output_file)
    except Exception as e:
        print(f"Error during anonymization: {e}")


if __name__ == "__main__":
    main()