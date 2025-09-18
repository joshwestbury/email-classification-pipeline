#!/usr/bin/env python3
"""
PII Detection Script for Collection Emails

This script analyzes cleaned email messages to identify potential
Personally Identifiable Information (PII) that may need anonymization.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """Represents a PII match found in text."""
    type: str
    value: str
    confidence: str  # high, medium, low
    start_pos: int
    end_pos: int
    context: str  # surrounding text for verification


class PIIDetector:
    """Detects various types of PII in text content."""

    def __init__(self):
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, str]]]:
        """Compile regex patterns for different PII types."""
        return {
            'email': [
                (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), 'high'),
            ],
            'phone': [
                # US/International phone formats
                (re.compile(r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'), 'high'),
                (re.compile(r'\+?61[-.\s]?\(?[0-9]{1,2}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'), 'high'),  # Australian
                (re.compile(r'\b\d{10,15}\b'), 'low'),  # Generic long number sequences
            ],
            'address': [
                # Street addresses
                (re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Place|Pl)\b', re.IGNORECASE), 'medium'),
                # Postal codes
                (re.compile(r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b'), 'medium'),  # UK
                (re.compile(r'\b\d{5}(?:-\d{4})?\b'), 'low'),  # US ZIP
                (re.compile(r'\b\d{4}\b'), 'low'),  # Australian postcodes
            ],
            'account_number': [
                # Bank account patterns (be conservative to avoid false positives)
                (re.compile(r'\b(?:Account\s*(?:Number|No\.?|#):?\s*)?(\d{8,20})\b', re.IGNORECASE), 'medium'),
                (re.compile(r'\b(?:BSB:?\s*)?(\d{3}[-\s]?\d{3})\b'), 'medium'),  # Australian BSB
            ],
            'invoice_reference': [
                # Invoice numbers (lower priority as they might be legitimate business refs)
                (re.compile(r'\b(?:Invoice|INV)[\s#]*([A-Z0-9]{6,20})\b', re.IGNORECASE), 'low'),
                (re.compile(r'\bPO\s*(?:Box|Number)?\s*([A-Z0-9]{6,15})\b', re.IGNORECASE), 'low'),
            ],
            'credit_card': [
                # Credit card patterns with Luhn check
                (re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), 'high'),
            ],
            'ssn': [
                # US SSN format
                (re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'), 'medium'),
            ],
            'tax_id': [
                # ABN (Australian Business Number)
                (re.compile(r'\b(?:ABN:?\s*)?(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b', re.IGNORECASE), 'medium'),
                # ACN (Australian Company Number)
                (re.compile(r'\b(?:ACN:?\s*)?(\d{3}\s?\d{3}\s?\d{3})\b', re.IGNORECASE), 'medium'),
            ],
            'name_patterns': [
                # Person names (very conservative, high false positive risk)
                (re.compile(r'\bMr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'), 'low'),
                (re.compile(r'\bMs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'), 'low'),
                (re.compile(r'\bMrs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'), 'low'),
                (re.compile(r'\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'), 'low'),
            ]
        }

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        digits = [int(d) for d in card_number if d.isdigit()]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10 == 0

    def _get_context(self, text: str, start: int, end: int, context_length: int = 50) -> str:
        """Extract context around a match."""
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)
        return text[context_start:context_end].strip()

    def detect_pii(self, text: str, email_id: int = None) -> List[PIIMatch]:
        """Detect PII in given text."""
        if not text or not isinstance(text, str):
            return []

        matches = []

        for pii_type, patterns in self.patterns.items():
            for pattern, confidence in patterns:
                for match in pattern.finditer(text):
                    value = match.group(1) if match.groups() else match.group(0)

                    # Special validation for credit cards
                    if pii_type == 'credit_card':
                        if not self._luhn_check(value):
                            continue

                    # Skip common false positives
                    if self._is_likely_false_positive(pii_type, value, text, match.start()):
                        continue

                    context = self._get_context(text, match.start(), match.end())

                    matches.append(PIIMatch(
                        type=pii_type,
                        value=value,
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=context
                    ))

        return matches

    def _is_likely_false_positive(self, pii_type: str, value: str, text: str, position: int) -> bool:
        """Check if a match is likely a false positive."""
        value_lower = value.lower()

        # Skip common business/system identifiers
        if pii_type == 'account_number':
            # Skip if it looks like a phone number or date
            if len(value) == 10 and (value.startswith('04') or value.startswith('02')):
                return True
            if re.match(r'\d{2,4}/\d{2,4}/\d{2,4}', value):
                return True

        # Skip invoice numbers that are clearly business references
        if pii_type == 'invoice_reference':
            if len(value) < 6:  # Too short to be meaningful
                return True

        # Skip email addresses that are clearly business/support addresses
        if pii_type == 'email':
            business_indicators = [
                'support', 'billing', 'info', 'admin', 'sales', 'accounts',
                'noreply', 'no-reply', 'contact', 'help', 'service'
            ]
            if any(indicator in value_lower for indicator in business_indicators):
                return True

        # Skip phone numbers in business signatures
        if pii_type == 'phone':
            # Look for business context indicators
            context = text[max(0, position-100):position+100].lower()
            business_indicators = ['phone:', 'mobile:', 'tel:', 'office:', 'direct:']
            if any(indicator in context for indicator in business_indicators):
                return True

        return False


def analyze_emails(input_file: Path) -> Dict[str, Any]:
    """Analyze emails and identify those containing PII."""
    detector = PIIDetector()

    print(f"Loading emails from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    print(f"Analyzing {len(emails)} emails for PII...")

    pii_emails = []
    pii_stats = {
        'total_emails': len(emails),
        'emails_with_pii': 0,
        'pii_by_type': {},
        'confidence_levels': {'high': 0, 'medium': 0, 'low': 0}
    }

    for i, email in enumerate(emails):
        if i % 500 == 0:
            print(f"Processed {i}/{len(emails)} emails...")

        email_id = email.get('id')
        subject = email.get('subject', '')
        message = email.get('message', '')

        # Check both subject and message for PII (handle None values)
        subject_matches = detector.detect_pii(subject or '', email_id)
        message_matches = detector.detect_pii(message or '', email_id)

        all_matches = subject_matches + message_matches

        if all_matches:
            pii_emails.append({
                'id': email_id,
                'subject': subject,
                'message': message,
                'pii_detected': [
                    {
                        'type': match.type,
                        'value': match.value,
                        'confidence': match.confidence,
                        'context': match.context
                    }
                    for match in all_matches
                ]
            })

            # Update statistics
            pii_stats['emails_with_pii'] += 1

            for match in all_matches:
                pii_stats['pii_by_type'][match.type] = pii_stats['pii_by_type'].get(match.type, 0) + 1
                pii_stats['confidence_levels'][match.confidence] += 1

    print(f"Analysis complete. Found {len(pii_emails)} emails with potential PII.")

    return {
        'statistics': pii_stats,
        'emails_with_pii': pii_emails
    }


def main():
    """Main function to process emails and generate PII report."""
    input_file = Path("emails_cleaned.json")
    output_file = Path("emails_with_pii.json")

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        return

    try:
        # Analyze emails for PII
        results = analyze_emails(input_file)

        # Save results
        print(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        stats = results['statistics']
        print("\n" + "="*50)
        print("PII DETECTION SUMMARY")
        print("="*50)
        print(f"Total emails analyzed: {stats['total_emails']}")
        print(f"Emails with PII detected: {stats['emails_with_pii']}")
        print(f"Percentage with PII: {stats['emails_with_pii']/stats['total_emails']*100:.1f}%")
        print("\nPII Types Found:")
        for pii_type, count in sorted(stats['pii_by_type'].items()):
            print(f"  {pii_type}: {count}")
        print("\nConfidence Levels:")
        for level, count in stats['confidence_levels'].items():
            print(f"  {level}: {count}")

        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()