#!/usr/bin/env python3
"""
Email Anonymization Script for Master Email Threads

Anonymizes sensitive PII in the master email threads dataset while preserving
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
            'tax_id': 0,
            'name': 0,
            'company': 0
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
        """Anonymize phone numbers while preserving format."""
        phone = match.group(0)
        return self._generate_consistent_replacement(phone, 'phone')

    def _anonymize_address(self, match: re.Match) -> str:
        """Anonymize street addresses while preserving structure."""
        address = match.group(0)
        # Extract street type
        street_types = ['Street', 'St', 'Avenue', 'Ave', 'Road', 'Rd', 'Drive', 'Dr', 'Lane', 'Ln', 'Boulevard', 'Blvd', 'Place', 'Pl', 'Suite', 'Level', 'Floor', 'Unit']
        street_type = next((st for st in street_types if st.lower() in address.lower()), 'Street')

        self.replacement_counters['address'] += 1
        return f"[ADDRESS_{self.replacement_counters['address']} {street_type}]"

    def _anonymize_account_number(self, match: re.Match) -> str:
        """Anonymize account numbers."""
        account = match.group(0)
        # Skip invoice numbers (keep business context)
        if 'INV' in account or 'inv' in account:
            return account
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

        self.replacement_counters['tax_id'] += 1
        return f"[{id_type}_{self.replacement_counters['tax_id']}]"

    def anonymize_text_patterns(self, text: str) -> str:
        """Pattern-based anonymization for common PII types."""
        if not text:
            return text

        anonymized = text

        # Apply all anonymization rules
        for rule_name, rule in self.rules.items():
            anonymized = rule.pattern.sub(rule.replacement_func, anonymized)

        # Additional patterns
        # Anonymize postal codes
        anonymized = re.sub(r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b', '[POSTAL_CODE]', anonymized)
        anonymized = re.sub(r'\b\d{4,5}(?:-\d{4})?\b', '[POSTAL_CODE]', anonymized)

        # Anonymize company names (but preserve some business context)
        company_patterns = [
            r'\b[A-Z][a-z]+\s+(?:Pty\s+Ltd|LLC|Inc|Corporation|Corp|Limited|Ltd)\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Pty\s+Ltd|LLC|Inc|Corporation|Corp|Limited|Ltd)\b'
        ]
        for pattern in company_patterns:
            matches = re.finditer(pattern, anonymized)
            for match in reversed(list(matches)):  # Process from end to start
                company_name = match.group(0)
                # Don't anonymize Litera (needed for classification)
                if 'Litera' not in company_name:
                    replacement = self._generate_consistent_replacement(company_name, 'company')
                    start, end = match.span()
                    anonymized = anonymized[:start] + replacement + anonymized[end:]

        # Anonymize person names in signatures
        name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b(?=\s*(?:\n|\r\n).*(?:Manager|Director|Specialist|Executive|Assistant))',
        ]
        for pattern in name_patterns:
            matches = re.finditer(pattern, anonymized, re.MULTILINE)
            for match in reversed(list(matches)):
                name = match.group(0)
                replacement = self._generate_consistent_replacement(name, 'name')
                start, end = match.span()
                anonymized = anonymized[:start] + replacement + anonymized[end:]

        return anonymized


def anonymize_master_email_threads(input_file: Path, output_file: Path):
    """Anonymize the master email threads dataset."""
    print(f"Loading master email threads from {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    anonymizer = EmailAnonymizer()

    total_threads = len(data['email_threads'])
    total_emails = data['metadata']['final_total_emails']
    anonymized_emails_count = 0

    print(f"Anonymizing {total_emails} emails across {total_threads} threads...")

    # Process each email thread
    for thread_idx, thread in enumerate(data['email_threads']):
        if thread_idx % 250 == 0:
            print(f"  Processed {thread_idx}/{total_threads} threads...")

        # Anonymize thread subject
        thread['subject'] = anonymizer.anonymize_text_patterns(thread['subject'])

        # Process each email in the thread
        for email in thread['emails']:
            original_content = email['content']

            # Anonymize email content
            anonymized_content = anonymizer.anonymize_text_patterns(original_content)

            # Update email record
            email['content'] = anonymized_content
            email['content_anonymized'] = original_content != anonymized_content

            if email['content_anonymized']:
                anonymized_emails_count += 1

            # Anonymize signature info if present
            if 'signature_info' in email:
                sig_info = email['signature_info']
                if sig_info.get('email') and '@litera.com' not in sig_info['email']:
                    sig_info['email'] = anonymizer.anonymize_text_patterns(sig_info['email'])
                if sig_info.get('phone'):
                    sig_info['phone'] = anonymizer.anonymize_text_patterns(sig_info['phone'])

    # Update metadata
    data['metadata']['anonymized'] = True
    data['metadata']['anonymization_date'] = '2024-09-18'
    data['metadata']['emails_with_anonymized_content'] = anonymized_emails_count
    data['metadata']['anonymization_rate'] = f"{anonymized_emails_count/total_emails*100:.1f}%"

    print(f"Saving anonymized dataset to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Generate anonymization report
    print(f"\n{'='*50}")
    print("ANONYMIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total email threads: {total_threads}")
    print(f"Total individual emails: {total_emails}")
    print(f"Emails with anonymized content: {anonymized_emails_count}")
    print(f"Anonymization rate: {anonymized_emails_count/total_emails*100:.1f}%")
    print(f"\nReplacement counts:")
    for pii_type, count in anonymizer.replacement_counters.items():
        if count > 0:
            print(f"  {pii_type}: {count}")

    print(f"\nAnonymized dataset saved to: {output_file}")


def main():
    """Main function to anonymize master email threads dataset."""
    input_file = Path("master_email_threads.json")
    output_file = Path("master_email_threads_anonymized.json")

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return

    try:
        anonymize_master_email_threads(input_file, output_file)
    except Exception as e:
        print(f"Error during anonymization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()