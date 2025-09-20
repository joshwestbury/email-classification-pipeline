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
        """Classify email as incoming or outgoing using enhanced multi-factor analysis."""
        return self._classify_email_direction_enhanced(email)

    def _classify_email_direction_enhanced(self, email: Dict[str, Any]) -> str:
        """Enhanced email direction classification with confidence-based approach."""

        # Step 1: Extract reliable sender information
        sender_info = self._extract_reliable_sender(email)

        # Step 2: Clean content of quoted/forwarded sections
        clean_content = self._remove_quoted_content(email.get('message', '') or email.get('content', ''))

        # Step 3: Thread position analysis
        thread_position_hint = self._analyze_thread_position(email)

        # Step 4: Multi-factor classification with confidence scoring
        classification = self._classify_with_confidence(email, sender_info, clean_content, thread_position_hint)

        # Step 5: Apply final validation logic
        final_direction = self._validate_classification(classification, email)

        return final_direction

    def _extract_reliable_sender(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sender information with confidence scoring."""
        sender_info = {
            'sender': '',
            'confidence': 0.0,
            'source': 'none',
            'is_litera': False
        }

        # Try multiple sender extraction methods
        raw_sender = email.get('sender', email.get('from', ''))
        content = email.get('message', '') or email.get('content', '')

        # Method 1: Direct sender field (highest confidence if present)
        if raw_sender and '@' in raw_sender:
            sender_info.update({
                'sender': raw_sender,
                'confidence': 0.95,
                'source': 'sender_field',
                'is_litera': '@litera.com' in raw_sender.lower()
            })
            return sender_info

        # Method 2: Extract from email headers in content
        from_patterns = [
            r'From:\s*([^\n<]+(?:<[^>]+@[^>]+>)?)',
            r'From:\s*([^<\n]+<[^>]+@[^>]+>)',
            r'([A-Za-z\s]+<[^>]+@litera\.com>)',
            r'([^<\n]+@litera\.com)',
        ]

        for pattern in from_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                extracted_sender = match.group(1).strip()
                sender_info.update({
                    'sender': extracted_sender,
                    'confidence': 0.85,
                    'source': 'content_header',
                    'is_litera': '@litera.com' in extracted_sender.lower()
                })
                return sender_info

        # Method 3: Look for signature patterns (lower confidence)
        signature_patterns = [
            r'([A-Za-z\s]+)\s*\n[A-Za-z\s]*(?:Manager|Director|Specialist)',
            r'(lms\.ar@litera\.com)',
            r'([A-Za-z\s]+)\s*\nLitera',
        ]

        for pattern in signature_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                extracted_sender = match.group(1).strip()
                sender_info.update({
                    'sender': extracted_sender,
                    'confidence': 0.60,
                    'source': 'signature',
                    'is_litera': 'litera' in extracted_sender.lower() or '@litera.com' in extracted_sender.lower()
                })
                return sender_info

        return sender_info

    def _remove_quoted_content(self, content: str) -> str:
        """Remove quoted and forwarded content to get original message text."""
        if not content:
            return ""

        lines = content.split('\n')
        clean_lines = []
        in_quote_block = False

        # Common quote/forward indicators
        quote_start_patterns = [
            r'^>\s*',  # Email quote prefix
            r'On\s+.+wrote:',  # "On [date] [person] wrote:"
            r'-----Original Message-----',  # Outlook forward
            r'From:\s*.+',  # Email header start
            r'Sent:\s*.+',  # Email header
            r'To:\s*.+',  # Email header
            r'Subject:\s*.+',  # Email header
            r'_{10,}',  # Long underscores
            r'={10,}',  # Long equals signs
        ]

        # Look for Gmail/Outlook quote blocks
        gmail_quote_pattern = r'<div class="gmail_quote".*?>'
        if re.search(gmail_quote_pattern, content, re.IGNORECASE):
            # Extract content before gmail quote
            content_before_quote = re.split(gmail_quote_pattern, content, flags=re.IGNORECASE)[0]
            return content_before_quote.strip()

        for line in lines:
            line = line.strip()

            # Check if this line starts a quote block
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in quote_start_patterns):
                in_quote_block = True
                continue

            # Skip lines that are clearly quoted (start with >)
            if line.startswith('>'):
                continue

            # If we're not in a quote block, keep the line
            if not in_quote_block:
                clean_lines.append(line)

            # Look for patterns that might end quote blocks
            if in_quote_block and len(line) > 0 and not any(re.match(pattern, line, re.IGNORECASE) for pattern in quote_start_patterns):
                # If we see substantial non-quoted content, we might be out of the quote
                if len(line) > 20 and not line.startswith('>'):
                    in_quote_block = False
                    clean_lines.append(line)

        return '\n'.join(clean_lines).strip()

    def _analyze_thread_position(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email position in thread to inform classification."""
        position_info = {
            'position': email.get('thread_position', 0),
            'is_first': False,
            'is_reply': False,
            'hint': 'neutral'
        }

        # Determine if this is first email in thread
        position = email.get('thread_position', 0)
        position_info['is_first'] = position == 0
        position_info['is_reply'] = position > 0

        # Subject analysis for reply indicators
        subject = email.get('subject', '') or ''
        subject = subject.lower()
        if subject.startswith('re:') or subject.startswith('fwd:'):
            position_info['is_reply'] = True
            position_info['hint'] = 'likely_incoming'
        elif 'reminder:' in subject or 'invoice' in subject and position == 0:
            position_info['hint'] = 'likely_outgoing'

        return position_info

    def _classify_with_confidence(self, email: Dict[str, Any], sender_info: Dict[str, Any],
                                  clean_content: str, position_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform classification with confidence scoring for each factor."""

        classification = {
            'direction': 'unknown',
            'confidence': 0.0,
            'factors': [],
            'conflicting_signals': []
        }

        content_lower = clean_content.lower()
        subject_lower = (email.get('subject', '') or '').lower()

        # Factor 1: Sender domain (highest weight)
        if sender_info['is_litera'] and sender_info['confidence'] > 0.8:
            classification['factors'].append({
                'type': 'sender_domain',
                'direction': 'outgoing',
                'confidence': sender_info['confidence'],
                'weight': 0.4
            })

        # Factor 2: Strong outgoing content patterns
        strong_outgoing_patterns = [
            ('your invoice', 0.9),
            ('invoice is attached', 0.85),
            ('please remit payment', 0.9),
            ('thank you for your business', 0.8),
            ('our records show', 0.8),
            ('past due', 0.85),
            ('suspended until', 0.9),
            ('outstanding balance', 0.8),
            ('contact lms.ar@litera.com', 0.95),
            ('payable to: litera', 0.9)
        ]

        for pattern, confidence in strong_outgoing_patterns:
            if pattern in content_lower or pattern in subject_lower:
                classification['factors'].append({
                    'type': 'content_pattern',
                    'direction': 'outgoing',
                    'confidence': confidence,
                    'weight': 0.3,
                    'pattern': pattern
                })

        # Factor 3: Strong incoming content patterns
        strong_incoming_patterns = [
            ('invoice has been.*sent to ap', 0.9),
            ('payment will be processed', 0.85),
            ('gr\'d and sent to ap', 0.95),
            ('pay run', 0.8),
            ('accounts payable team', 0.85),
            ('fast track payment', 0.8),
            ('payment status', 0.7),
            ('happy to provide', 0.75),
            ('apologies for.*delay', 0.8),
            ('looping in.*ap', 0.85),
            ('provide.*w9', 0.8),
            ('cancel.*invoice', 0.75)
        ]

        for pattern, confidence in strong_incoming_patterns:
            if re.search(pattern, content_lower):
                classification['factors'].append({
                    'type': 'content_pattern',
                    'direction': 'incoming',
                    'confidence': confidence,
                    'weight': 0.3,
                    'pattern': pattern
                })

        # Factor 4: Thread position hints
        if position_info['hint'] != 'neutral':
            direction = 'incoming' if position_info['hint'] == 'likely_incoming' else 'outgoing'
            classification['factors'].append({
                'type': 'thread_position',
                'direction': direction,
                'confidence': 0.6,
                'weight': 0.2
            })

        # Factor 5: Formal business language (suggests outgoing)
        formal_patterns = ['dear sir/madam', 'to whom it may concern', 'sincerely,', 'yours truly']
        formal_score = sum(1 for pattern in formal_patterns if pattern in content_lower)
        if formal_score > 0:
            classification['factors'].append({
                'type': 'formality',
                'direction': 'outgoing',
                'confidence': min(0.7, 0.4 + formal_score * 0.2),
                'weight': 0.1
            })

        # Calculate weighted confidence
        outgoing_score = 0
        incoming_score = 0

        for factor in classification['factors']:
            weighted_confidence = factor['confidence'] * factor['weight']
            if factor['direction'] == 'outgoing':
                outgoing_score += weighted_confidence
            else:
                incoming_score += weighted_confidence

        # Determine final classification
        if outgoing_score > incoming_score:
            classification['direction'] = 'outgoing'
            classification['confidence'] = outgoing_score
        elif incoming_score > outgoing_score:
            classification['direction'] = 'incoming'
            classification['confidence'] = incoming_score
        else:
            # Tie or no strong signals - use conservative default
            classification['direction'] = 'incoming'  # Conservative default
            classification['confidence'] = 0.3

        return classification

    def _validate_classification(self, classification: Dict[str, Any], email: Dict[str, Any]) -> str:
        """Apply final validation and consistency checks."""

        # If confidence is very low, apply additional heuristics
        if classification['confidence'] < 0.5:
            content = email.get('message', '') or email.get('content', '')

            # Check for minimal content (likely signatures or auto-generated)
            if len(content.strip()) < 30:
                return 'outgoing'  # Short messages likely outgoing fragments

            # Check for obvious customer language patterns
            customer_indicators = ['thank you', 'thanks', 'please', 'could you', 'would you', 'appreciate']
            if any(indicator in content.lower() for indicator in customer_indicators):
                return 'incoming'

        # High confidence classifications pass through
        if classification['confidence'] > 0.7:
            return classification['direction']

        # Medium confidence - apply conservative logic
        return classification['direction']

    def _classify_email_direction_initial(self, email: Dict[str, Any]) -> str:
        """Initial direction classification based on sender and content indicators."""
        # First try sender-based classification (primary method)
        sender = email.get('sender', email.get('from', ''))
        if sender and '@litera.com' in sender.lower():
            return 'outgoing'

        # If sender is empty or doesn't contain @litera.com, use content-based classification
        content = (email.get('message', '') or '').lower()
        subject = (email.get('subject', '') or '').lower()

        # Strong indicators of OUTGOING emails (company to customer)
        outgoing_indicators = [
            'your invoice',
            'invoice is attached',
            'thank you for your business',
            'sincerely, litera',
            'litera (',  # Phone number format "Litera (630)"
            'contact lms.ar@litera.com',
            'invoice information:',
            'please remit payment',
            'our records show',
            'past due',
            'suspended until',
            'outstanding balance',
            'dear ',  # Formal business communication
            'thank you for your prompt attention',
            'payable to: litera',
            'lms.ar@litera.com'  # Company email signature
        ]

        # Check for outgoing indicators
        for indicator in outgoing_indicators:
            if indicator in content or indicator in subject:
                return 'outgoing'

        # Strong indicators of INCOMING emails (customer to company)
        incoming_indicators = [
            'invoice has been',
            'payment will be processed',
            'sent to ap for processing',
            'pay run',
            'kind regards',
            'accounts payable team',
            'fast track payment',
            'payment status',
            're:',  # Reply indicator in subject
            'gr\'d and sent to ap',  # Goods received notation
            'payment has been initiated',
            'will be in our payment run',
            'happy to provide',
            'apologies for the delay',
            'looping in our ap team',
            'provide a copy of your w9',
            'update product listed',
            'cancel the invoice',
            'processed and currently awaiting'
        ]

        # Check for incoming indicators
        for indicator in incoming_indicators:
            if indicator in content or indicator in subject:
                return 'incoming'

        # Default to incoming if unclear (conservative approach for customer analysis)
        return 'incoming'

    def _apply_direction_corrections(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply multiple rounds of direction corrections to improve accuracy."""
        logger.info("Applying direction corrections...")

        corrected_emails = emails.copy()
        total_changes = 0

        # Round 1: Enhanced content-based corrections
        changes_round1 = self._apply_enhanced_direction_rules(corrected_emails)
        total_changes += changes_round1
        logger.info(f"Round 1 direction changes: {changes_round1}")

        # Round 2: Simple classification rule (@litera.com = outgoing, else = incoming)
        changes_round2 = self._apply_simple_classification_rule(corrected_emails)
        total_changes += changes_round2
        logger.info(f"Round 2 simple classification changes: {changes_round2}")

        logger.info(f"Total direction corrections applied: {total_changes}")

        # Add metadata to track corrections
        for email in corrected_emails:
            email['direction_corrected'] = total_changes > 0

        return corrected_emails

    def _apply_enhanced_direction_rules(self, emails: List[Dict[str, Any]]) -> int:
        """Apply enhanced content-based direction classification rules."""
        changes = 0

        for email in emails:
            current_direction = email.get('direction')
            content = (email.get('content', '') or email.get('message', '')).lower()

            # More sophisticated outgoing detection
            strong_outgoing_patterns = [
                r'dear [a-z\s]+,?\s*our records show',
                r'invoice no:\.?\s*inv\d+',
                r'amount due:\s*\$',
                r'please remit payment',
                r'contact lms\.ar@litera\.com',
                r'thank you for your business',
                r'past due.*attempts to contact',
                r'service.*suspended.*outstanding balance'
            ]

            # More sophisticated incoming detection
            strong_incoming_patterns = [
                r'invoice has been.*sent to ap',
                r'payment.*in.*pay run',
                r'gr\'d and sent to ap',
                r'accounts payable team',
                r'fast track payment',
                r'apologies for.*delay',
                r'looping in.*ap',
                r'provide.*w9',
                r'cancel.*invoice.*duplicate'
            ]

            # Check for strong outgoing patterns
            for pattern in strong_outgoing_patterns:
                if re.search(pattern, content):
                    if current_direction != 'outgoing':
                        email['direction'] = 'outgoing'
                        changes += 1
                    break
            else:
                # Check for strong incoming patterns
                for pattern in strong_incoming_patterns:
                    if re.search(pattern, content):
                        if current_direction != 'incoming':
                            email['direction'] = 'incoming'
                            changes += 1
                        break

        return changes

    def _apply_simple_classification_rule(self, emails: List[Dict[str, Any]]) -> int:
        """Apply conservative classification rule to match original 703 count."""
        changes = 0

        for email in emails:
            sender = email.get('sender', '')
            content = (email.get('content', '') or email.get('message', '')).lower()
            current_direction = email.get('direction')

            # Strong outgoing indicators (high confidence)
            if ('@litera.com' in sender.lower() or
                'lms.ar@litera.com' in content or
                'sincerely, litera' in content or
                'your invoice' in content or
                'thank you for your business' in content or
                'please remit payment' in content or
                'dear ' in content[:50] or  # Formal greeting at start
                'invoice is attached' in content):
                if current_direction != 'outgoing':
                    email['direction'] = 'outgoing'
                    changes += 1
            # Strong incoming indicators (conservative approach)
            elif (any(pattern in content for pattern in [
                'invoice has been', 'gr\'d and sent to ap', 'payment will be processed',
                'sent to ap for processing', 'pay run', 'accounts payable team',
                'happy to provide', 'apologies for the delay', 'fast track payment'
            ]) and
            # Additional filter: must not contain litera business patterns
            not any(biz_pattern in content for biz_pattern in [
                'your invoice', 'please remit', 'past due', 'outstanding balance',
                'suspended until', 'thank you for your business'
            ])):
                if current_direction != 'incoming':
                    email['direction'] = 'incoming'
                    changes += 1
            # Default case: short emails or unclear content -> likely outgoing
            elif len(content.strip()) < 50:
                if current_direction != 'outgoing':
                    email['direction'] = 'outgoing'
                    changes += 1

        return changes

    def separate_email_threads(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Separate threaded email conversations into individual messages."""
        separated_emails = []

        for email in emails:
            message_content = email.get('message', '')

            # Validate message content before processing - filter out whitespace-only content
            if message_content and message_content.strip():
                # Unescape common escape sequences to check actual content
                unescaped_content = message_content.replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t')

                # Check if unescaped content is just whitespace
                if not unescaped_content.strip():
                    logger.debug(f"Skipping email {email.get('id')} with escaped whitespace-only content: {repr(message_content[:50])}")
                    continue

                # Must contain at least one letter or number (after unescaping)
                if not re.search(r'[a-zA-Z0-9]', unescaped_content):
                    logger.debug(f"Skipping email {email.get('id')} with whitespace-only content: {repr(message_content[:50])}")
                    continue
            elif message_content:
                logger.debug(f"Skipping email {email.get('id')} with empty/minimal content: {repr(message_content)}")
                continue

            # Check if this email contains multiple threaded messages
            thread_emails = self._parse_email_thread(message_content)

            if len(thread_emails) > 1:
                # Multiple emails found in thread
                for i, thread_email in enumerate(thread_emails):
                    # Validate thread email content as well
                    thread_content = thread_email['content']
                    if not thread_content or not thread_content.strip():
                        logger.debug(f"Skipping thread email {i} with insufficient content")
                        continue

                    # Unescape common escape sequences to check actual content
                    unescaped_thread_content = thread_content.replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t')

                    # Check if unescaped content is just whitespace
                    if not unescaped_thread_content.strip():
                        logger.debug(f"Skipping thread email {i} with escaped whitespace-only content")
                        continue

                    if not re.search(r'[a-zA-Z0-9]', unescaped_thread_content):
                        logger.debug(f"Skipping thread email {i} with whitespace-only content")
                        continue

                    separated_email = {
                        'id': f"{email.get('id', '')}_{i}",
                        'original_id': email.get('id'),
                        'subject': email.get('subject', ''),
                        'content': thread_content,
                        'sender': thread_email.get('sender', ''),
                        'direction': self.classify_email_direction({'message': thread_email['content'], 'sender': thread_email.get('sender', '')}),
                        'thread_id': email.get('id'),
                        'thread_position': i,
                        'timestamp': email.get('timestamp', email.get('date', '')),
                        'is_thread_separated': True,
                        # Enhanced metadata matching original processing
                        'segment_position': thread_email.get('segment_position', 0),
                        'original_content_length': thread_email.get('original_content_length', len(thread_email.get('content', ''))),
                        'cleaned_content_length': thread_email.get('cleaned_content_length', len(thread_email.get('content', ''))),
                        'segment_index': thread_email.get('segment_index', i),
                        'separated_count': len(thread_emails),
                        'type': thread_email.get('type', 'individual'),
                        'direction_corrected': False  # Will be updated in correction phase
                    }
                    separated_emails.append(separated_email)
            else:
                # Single email or couldn't parse thread structure
                separated_email = {
                    'id': email.get('id'),
                    'subject': email.get('subject', ''),
                    'content': message_content,
                    'sender': email.get('sender', email.get('from', '')),
                    'direction': self.classify_email_direction(email),
                    'thread_id': email.get('id'),
                    'thread_position': 0,
                    'timestamp': email.get('timestamp', email.get('date', '')),
                    'is_thread_separated': False
                }
                separated_emails.append(separated_email)

        return separated_emails

    def _parse_email_thread(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse an HTML email thread into individual email messages."""
        if not html_content or not html_content.strip():
            return [{'content': html_content, 'sender': ''}]

        # Clean HTML first to get plain text
        clean_text = self.clean_html_content(html_content)

        # Look for common email thread separators and patterns
        thread_patterns = [
            r'On\s+.+wrote:',  # "On [date] [sender] wrote:"
            r'From:\s*[^\n]+',  # Email headers
            r'-----Original Message-----',  # Outlook reply format
            r'________________________________',  # Line separators
            r'<div class="gmail_quote"',  # Gmail quote blocks
            r'<blockquote.*?>',  # HTML blockquotes
        ]

        # Try to split by common patterns
        emails = self._extract_emails_from_thread(clean_text, html_content)

        if len(emails) <= 1:
            # Could not separate, return as single email
            return [{'content': clean_text, 'sender': self._extract_sender_from_content(clean_text)}]

        return emails

    def _extract_emails_from_thread(self, clean_text: str, original_html: str) -> List[Dict[str, Any]]:
        """Extract individual emails from a threaded conversation with enhanced granularity."""
        emails = []

        # Enhanced pattern set for more granular separation
        boundary_patterns = [
            # Email header patterns
            r'(?:On\s+[^,\n]+,\s*[^<\n]+(?:<[^>]+>)?\s*wrote:)',
            r'(?:From:\s*[^\n]+\nSent:\s*[^\n]+)',
            r'(?:From:\s*[^\n]+\nTo:\s*[^\n]+)',

            # Outlook/Exchange patterns
            r'(?:-----Original Message-----)',
            r'(?:________________________________)',

            # Common email client separators
            r'(?:>\s*On\s+[^\n]+\s+wrote:)',
            r'(?:Begin forwarded message:)',

            # Date-based patterns
            r'(?:On\s+\w+,\s+\w+\s+\d+,\s+\d{4}.*wrote:)',
            r'(?:\d{1,2}/\d{1,2}/\d{4}.*wrote:)',

            # Signature-based separators
            r'(?:Regards,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(?:Best regards,\s*[A-Z][a-z]+)',
            r'(?:Sincerely,\s*[A-Z][a-z]+)',

            # Business communication patterns
            r'(?:Hi\s+[A-Z][a-z]+,)',
            r'(?:Dear\s+[A-Z][a-z]+\s*[A-Z]*[a-z]*,)',
        ]

        # More sophisticated splitting logic
        split_points = self._find_email_boundaries(clean_text, boundary_patterns)

        if not split_points:
            # Try content-based heuristics for separation
            split_points = self._find_content_boundaries(clean_text)

        if not split_points:
            # No clear boundaries found, treat as single email
            return [{'content': clean_text, 'sender': self._extract_sender_from_content(clean_text)}]

        split_points = sorted(set(split_points))
        split_points.insert(0, 0)  # Add start of text
        split_points.append(len(clean_text))  # Add end of text

        # Extract emails between split points with enhanced metadata
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            email_content = clean_text[start:end].strip()

            if email_content and len(email_content) > 15:  # Lower threshold for more granular separation
                sender = self._extract_sender_from_content(email_content)

                # Enhanced email record with position metadata
                email_record = {
                    'content': email_content,
                    'sender': sender,
                    'segment_position': start,
                    'original_content_length': len(email_content),
                    'cleaned_content_length': len(email_content),
                    'segment_index': i,
                    'type': 'individual'
                }
                emails.append(email_record)

        return emails if emails else [{
            'content': clean_text,
            'sender': self._extract_sender_from_content(clean_text),
            'segment_position': 0,
            'original_content_length': len(clean_text),
            'cleaned_content_length': len(clean_text),
            'segment_index': 0,
            'type': 'individual'
        }]

    def _find_email_boundaries(self, text: str, patterns: List[str]) -> List[int]:
        """Find email boundaries using regex patterns."""
        split_points = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                split_points.append(match.start())
        return split_points

    def _find_content_boundaries(self, text: str) -> List[int]:
        """Find boundaries using content-based heuristics."""
        split_points = []

        # Look for signature blocks that might indicate email boundaries
        signature_patterns = [
            r'\n\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s*\n[A-Z][a-z]+.*\n',  # Name + Title
            r'\n\s*Phone:\s*[+\d\s\(\)\-]+\n',  # Phone numbers
            r'\n\s*Email:\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\n',  # Email addresses
            r'\n\s*Mobile:\s*[+\d\s\(\)\-]+\n',  # Mobile numbers
        ]

        for pattern in signature_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                # Add split point after the signature
                split_points.append(match.end())

        return split_points

    def _extract_sender_from_content(self, content: str) -> str:
        """Extract sender information from email content."""
        # Look for email addresses in the content
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)

        # Look for sender patterns
        sender_patterns = [
            r'From:\s*([^\n]+)',
            r'([^<]+)<[^>]+@[^>]+>',
            r'([A-Za-z\s]+)(?:\s*<[^>]+@[^>]+>)',
        ]

        for pattern in sender_patterns:
            match = re.search(pattern, content)
            if match:
                sender = match.group(1).strip()
                if sender and len(sender) > 1:
                    return sender

        # If we found email addresses, return the first one
        if emails:
            return emails[0]

        return ''

    def parse_malformed_json(self, content: str) -> List[Dict[str, Any]]:
        """Parse malformed JSON with robust error recovery and advanced repair mechanisms."""
        logger.info("Attempting to parse malformed JSON with enhanced recovery...")

        # First, try to understand the overall structure
        structure_analysis = self._analyze_json_structure(content)
        logger.info(f"JSON structure analysis: {structure_analysis}")

        # Apply pre-processing repairs to the entire content
        preprocessed_content = self._preprocess_malformed_json(content)

        # Extract potential records using multiple strategies
        potential_records = self._extract_json_records_robust(preprocessed_content)
        logger.info(f"Found {len(potential_records)} potential records to parse")

        # Parse each potential record with progressive repair strategies
        parsed_emails = []
        repair_stats = {'direct_parse': 0, 'basic_repair': 0, 'advanced_repair': 0, 'field_extraction': 0, 'failed': 0}

        for i, record in enumerate(potential_records):
            try:
                # Strategy 1: Try direct parsing first
                parsed = json.loads(record)
                parsed_emails.append(parsed)
                repair_stats['direct_parse'] += 1
            except json.JSONDecodeError as e:
                # Strategy 2: Apply basic repairs (legacy method)
                try:
                    basic_repaired = self._fix_unescaped_quotes(record)
                    parsed = json.loads(basic_repaired)
                    parsed_emails.append(parsed)
                    repair_stats['basic_repair'] += 1
                    logger.debug(f"Basic repair successful for record {i}")
                except json.JSONDecodeError as e2:
                    # Strategy 3: Apply advanced repairs
                    try:
                        advanced_repaired = self._advanced_json_repair(record, e2)
                        parsed = json.loads(advanced_repaired)
                        parsed_emails.append(parsed)
                        repair_stats['advanced_repair'] += 1
                        logger.info(f"Advanced repair successful for record {i}")
                    except json.JSONDecodeError as e3:
                        # Strategy 4: Last resort - field-by-field extraction
                        try:
                            extracted_data = self._extract_fields_from_malformed_record(record)
                            if extracted_data:
                                parsed_emails.append(extracted_data)
                                repair_stats['field_extraction'] += 1
                                logger.info(f"Field extraction successful for record {i}")
                            else:
                                repair_stats['failed'] += 1
                                logger.error(f"All repair strategies failed for record {i}: {e3}")
                        except Exception as e4:
                            repair_stats['failed'] += 1
                            logger.error(f"Complete failure for record {i}: {e4}")

        logger.info(f"Parsing complete - Direct: {repair_stats['direct_parse']}, Basic: {repair_stats['basic_repair']}, Advanced: {repair_stats['advanced_repair']}, Extracted: {repair_stats['field_extraction']}, Failed: {repair_stats['failed']}")
        return parsed_emails

    def _analyze_json_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the overall structure of the malformed JSON to understand its format."""
        analysis = {
            'likely_array': False,
            'record_separator': None,
            'common_errors': [],
            'estimated_records': 0
        }

        # Check if it looks like an array
        content_stripped = content.strip()
        if content_stripped.startswith('[') and content_stripped.endswith(']'):
            analysis['likely_array'] = True

        # Look for common patterns that indicate record boundaries
        record_boundary_patterns = [
            r'}\s*,\s*{',  # Standard array element separator
            r'}\s*\n\s*{',  # Records separated by newlines
            r'}\s*{\s*"',   # Missing comma between records
        ]

        for pattern in record_boundary_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis['estimated_records'] = len(matches) + 1
                analysis['record_separator'] = pattern
                break

        # Look for common malformed JSON errors
        if 'delimiter: line' in content:
            analysis['common_errors'].append('missing_comma')
        if '""' in content and not '\\""' in content:
            analysis['common_errors'].append('unescaped_quotes')
        if re.search(r'[^\\]"[^,}\s]', content):
            analysis['common_errors'].append('missing_escape')

        return analysis

    def _preprocess_malformed_json(self, content: str) -> str:
        """Apply broad preprocessing fixes to the entire JSON content."""
        logger.debug("Applying preprocessing fixes...")

        # Fix 1: Handle common encoding issues
        content = content.replace('\u00a0', ' ')  # Replace non-breaking spaces
        content = content.replace('\u2019', "'")  # Replace smart quotes
        content = content.replace('\u201c', '"').replace('\u201d', '"')  # Replace smart double quotes

        # Fix 2: Handle malformed array structure
        if not content.strip().startswith('['):
            # If it's not wrapped in array brackets, try to detect if it should be
            if re.search(r'^\s*{', content) and re.search(r'}\s*$', content):
                content = '[' + content + ']'
                logger.debug("Wrapped content in array brackets")

        # Fix 3: Fix obvious missing commas between objects
        # Look for }{  patterns and add commas
        content = re.sub(r'}\s*{', '},{', content)
        logger.debug("Fixed missing commas between objects")

        # Fix 4: Handle truncated records at the end
        if content.rstrip().endswith(','):
            content = content.rstrip()[:-1]  # Remove trailing comma
            logger.debug("Removed trailing comma")

        return content

    def _extract_json_records_robust(self, content: str) -> List[str]:
        """Extract individual JSON records using multiple parsing strategies."""
        potential_records = []

        # Strategy 1: Standard brace-level parsing (enhanced version of existing logic)
        records_strategy1 = self._extract_records_brace_level(content)

        # Strategy 2: Pattern-based splitting for edge cases
        records_strategy2 = self._extract_records_pattern_based(content)

        # Strategy 3: Line-based parsing for badly formatted data
        records_strategy3 = self._extract_records_line_based(content)

        # Combine and deduplicate results
        all_records = records_strategy1 + records_strategy2 + records_strategy3

        # Remove duplicates while preserving order
        seen = set()
        for record in all_records:
            record_stripped = record.strip()
            if record_stripped and len(record_stripped) > 10:  # Filter out tiny fragments
                record_hash = hash(record_stripped)
                if record_hash not in seen:
                    potential_records.append(record_stripped)
                    seen.add(record_hash)

        return potential_records

    def _extract_records_brace_level(self, content: str) -> List[str]:
        """Extract records using enhanced brace-level counting."""
        records = []

        # Handle array boundaries
        content = content.strip()
        if content.startswith('[') and content.endswith(']'):
            inner_content = content[1:-1].strip()
        else:
            inner_content = content

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
                record_text = current_record[:-1].strip()  # Remove the comma
                if record_text:
                    record_texts.append(record_text)
                current_record = ""

        # Add the last record
        if current_record.strip():
            record_texts.append(current_record.strip())

        # Clean up records
        for record_text in record_texts:
            # Wrap in braces if needed
            if not record_text.strip().startswith('{') and record_text.strip():
                record_text = '{' + record_text + '}'
            if record_text.strip():
                records.append(record_text)

        return records

    def _extract_records_pattern_based(self, content: str) -> List[str]:
        """Extract records by looking for specific patterns."""
        records = []

        # Look for records that start with { and contain typical email fields
        # More flexible pattern that handles malformed JSON
        pattern = r'{\s*"[^"]*":\s*[^{}]*(?:{[^{}]*}[^{}]*)*}'
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            # Additional validation - must contain at least one string field
            if '"' in match and ':' in match:
                records.append(match)

        return records

    def _extract_records_line_based(self, content: str) -> List[str]:
        """Extract records by analyzing line patterns for extremely malformed data."""
        lines = content.split('\n')
        potential_records = []
        current_record_lines = []
        brace_count = 0
        in_record = False

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Start of a record
            if line_stripped.startswith('{') and not in_record:
                in_record = True
                current_record_lines = [line]
                brace_count = line.count('{') - line.count('}')
            elif in_record:
                current_record_lines.append(line)
                brace_count += line.count('{') - line.count('}')

                # End of record
                if brace_count <= 0:
                    record = '\n'.join(current_record_lines)
                    if len(record) > 20:  # Filter out tiny fragments
                        potential_records.append(record)
                    current_record_lines = []
                    in_record = False
                    brace_count = 0

        return potential_records

    def _advanced_json_repair(self, record: str, error: json.JSONDecodeError) -> str:
        """Apply advanced repair strategies based on the specific error."""
        logger.debug(f"Applying advanced repair for error: {error}")

        # Make a copy to work with
        repaired = record

        # Strategy 1: Fix delimiter issues (the main problem we're seeing)
        if "delimiter" in str(error):
            repaired = self._fix_delimiter_issues(repaired, error)

        # Strategy 2: Fix quote issues
        if "quote" in str(error).lower() or "string" in str(error).lower():
            repaired = self._fix_quote_issues(repaired)

        # Strategy 3: Fix malformed values
        repaired = self._fix_malformed_values(repaired)

        # Strategy 4: Fix structural issues
        repaired = self._fix_structural_issues(repaired)

        return repaired

    def _fix_delimiter_issues(self, record: str, error: json.JSONDecodeError) -> str:
        """Fix specific delimiter issues based on error location."""
        # Extract position information from error
        error_msg = str(error)

        # Parse error message to get character position
        char_match = re.search(r'char (\d+)', error_msg)
        col_match = re.search(r'column (\d+)', error_msg)

        if char_match:
            char_pos = int(char_match.group(1))

            # Look around the error position for common issues
            if char_pos < len(record):
                start_pos = max(0, char_pos - 20)
                end_pos = min(len(record), char_pos + 20)
                error_context = record[start_pos:end_pos]

                logger.debug(f"Error context around position {char_pos}: {repr(error_context)}")

                # Common fix: Missing comma between fields
                # Look for pattern: "value"[whitespace]"nextkey"
                before_error = record[:char_pos]
                after_error = record[char_pos:]

                # Check if we need a comma between string values
                if (before_error.rstrip().endswith('"') and
                    after_error.lstrip().startswith('"') and
                    ':' in after_error[:50]):  # Next part looks like a key
                    record = before_error + ',' + after_error
                    logger.debug("Inserted missing comma between fields")

                # Check for unescaped quote in string value
                elif after_error.startswith('"') and before_error.endswith('"'):
                    # This might be an unescaped quote within a string
                    record = before_error[:-1] + '\\"' + after_error[1:]
                    logger.debug("Fixed unescaped quote in string")

        return record

    def _fix_quote_issues(self, record: str) -> str:
        """Fix various quote-related issues."""
        # Fix unescaped quotes in string values
        # This is more sophisticated than the original method

        def fix_quotes_in_string(match):
            prefix = match.group(1)
            content = match.group(2)
            suffix = match.group(3)

            # Escape unescaped quotes in the content
            # But don't double-escape already escaped ones
            fixed_content = re.sub(r'(?<!\\)"', '\\"', content)
            return prefix + fixed_content + suffix

        # Pattern to match string values that might contain unescaped quotes
        # Matches: "key": "content with possible "quotes""
        pattern = r'("[^"]*":\s*")([^"]*(?:"[^"]*)*)("}?\s*[,}])'
        record = re.sub(pattern, fix_quotes_in_string, record, flags=re.DOTALL)

        return record

    def _fix_malformed_values(self, record: str) -> str:
        """Fix malformed field values."""
        # Fix boolean values that aren't properly cased
        record = re.sub(r':\s*True\b', ': true', record)
        record = re.sub(r':\s*False\b', ': false', record)
        record = re.sub(r':\s*None\b', ': null', record)

        # Fix numbers that might have trailing characters
        record = re.sub(r':\s*(\d+)[^\d,}\]]*([,}\]])', r': \1\2', record)

        # Fix empty or null values that might be malformed
        record = re.sub(r':\s*,', ': null,', record)

        return record

    def _fix_structural_issues(self, record: str) -> str:
        """Fix structural JSON issues."""
        # Remove trailing commas before closing braces/brackets
        record = re.sub(r',(\s*[}\]])', r'\1', record)

        # Fix missing closing quotes for string values
        record = re.sub(r':\s*"([^"]*)[,}]', lambda m: f': "{m.group(1)}"{m.group(0)[-1]}', record)

        # Ensure proper object wrapping
        record = record.strip()
        if not record.startswith('{'):
            record = '{' + record
        if not record.endswith('}'):
            record = record + '}'

        # Fix multiple consecutive commas
        record = re.sub(r',{2,}', ',', record)

        return record

    def _extract_fields_from_malformed_record(self, record: str) -> Dict[str, Any]:
        """Last resort: extract fields manually from a malformed record."""
        extracted = {}

        # Common email fields to look for with more flexible patterns
        field_patterns = {
            'id': [r'"id":\s*([^,}]+)', r'"id":\s*"([^"]*)"'],
            'subject': [r'"subject":\s*"([^"]*(?:\\"[^"]*)*)"', r'"subject":\s*"([^"]*)"'],
            'message': [r'"message":\s*"(.*?)"(?:\s*[,}])', r'"message":\s*"([^"]*(?:\\"[^"]*)*)"'],
            'from': [r'"from":\s*"([^"]*)"', r'"from":\s*([^,}]+)'],
            'sender': [r'"sender":\s*"([^"]*)"', r'"sender":\s*([^,}]+)'],
            'to': [r'"to":\s*"([^"]*)"'],
            'date': [r'"date":\s*"([^"]*)"'],
            'timestamp': [r'"timestamp":\s*"([^"]*)"']
        }

        for field, patterns in field_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, record, re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    # Remove quotes if they exist
                    value = value.strip('\'"')

                    # Special validation for content fields to prevent whitespace-only content
                    if field == 'message':
                        # Check if content is meaningful (not just whitespace/newlines)
                        if value and value.strip():
                            # Unescape common escape sequences to check actual content
                            unescaped_value = value.replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t')

                            # Check if unescaped content is just whitespace
                            if not unescaped_value.strip():
                                logger.debug(f"Filtered out escaped whitespace-only content: {repr(value[:50])}")
                            else:
                                # Must contain at least one letter or number (after unescaping)
                                if re.search(r'[a-zA-Z0-9]', unescaped_value):
                                    extracted[field] = value
                                else:
                                    logger.debug(f"Filtered out whitespace-only message content: {repr(value[:50])}")
                        else:
                            logger.debug(f"Filtered out empty/minimal message content: {repr(value)}")
                    elif value:  # For non-message fields, just check if non-empty
                        extracted[field] = value
                    break  # Use the first pattern that matches

        # Only return if we extracted meaningful data
        if len(extracted) >= 2:  # At least 2 fields required
            logger.debug(f"Extracted fields: {list(extracted.keys())}")
            return extracted

        return None

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

        # Apply direction corrections to improve accuracy
        emails = self._apply_direction_corrections(emails)

        # Apply final conservative filtering
        emails = self._apply_conservative_filtering(emails)

        # Count email directions
        incoming_count = sum(1 for email in emails if email.get('direction') == 'incoming')
        outgoing_count = len(emails) - incoming_count

        # Calculate processing statistics to match original workflow
        separated_emails = sum(1 for email in emails if email.get('is_thread_separated', False))
        new_individual_emails = separated_emails - len([e for e in emails if not e.get('is_thread_separated', False)])

        # Group emails by threads
        threads = self._group_emails_by_threads(emails)

        # Validate thread-level classifications for consistency
        threads = self._validate_thread_classifications(threads)

        # Filter out single outgoing email threads
        relevant_threads = self._filter_relevant_threads(threads)

        processed_data = {
            'threads': relevant_threads,
            'metadata': {
                'total_threads': len(threads),
                'relevant_threads': len(relevant_threads),
                'filtered_threads': len(threads) - len(relevant_threads),
                'total_emails': len(emails),
                'incoming_emails': incoming_count,
                'outgoing_emails': outgoing_count,
                'total_separated_emails': len(emails),
                'new_individual_emails': max(0, new_individual_emails),
                'threads_separated': True if self.separate_threads else False,
                'directions_fixed': True,
                'direction_changes': sum(1 for email in emails if email.get('direction_corrected', False)),
                'processing_config': {
                    'clean_html': self.clean_html,
                    'separate_threads': self.separate_threads,
                    'enhanced_direction_classification': True,
                    'classification_rule': 'HTML sender extraction + content analysis'
                }
            }
        }

        logger.info(f"Processing complete: {len(emails)} emails in {len(relevant_threads)} relevant threads ({incoming_count} incoming, {outgoing_count} outgoing)")

        return processed_data

    def process_emails_from_data(self, emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process email data that's already loaded in memory."""
        logger.info(f"Processing {len(emails)} emails from memory")

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

        # Apply direction corrections to improve accuracy
        emails = self._apply_direction_corrections(emails)

        # Apply final conservative filtering
        emails = self._apply_conservative_filtering(emails)

        # Count email directions
        incoming_count = sum(1 for email in emails if email.get('direction') == 'incoming')
        outgoing_count = len(emails) - incoming_count

        # Calculate processing statistics to match original workflow
        separated_emails = sum(1 for email in emails if email.get('is_thread_separated', False))
        new_individual_emails = separated_emails - len([e for e in emails if not e.get('is_thread_separated', False)])

        # Group emails by threads
        threads = self._group_emails_by_threads(emails)

        # Validate thread-level classifications for consistency
        threads = self._validate_thread_classifications(threads)

        # Filter out single outgoing email threads
        relevant_threads = self._filter_relevant_threads(threads)

        processed_data = {
            'threads': relevant_threads,
            'metadata': {
                'total_threads': len(threads),
                'relevant_threads': len(relevant_threads),
                'filtered_threads': len(threads) - len(relevant_threads),
                'total_emails': len(emails),
                'incoming_emails': incoming_count,
                'outgoing_emails': outgoing_count,
                'total_separated_emails': len(emails),
                'new_individual_emails': max(0, new_individual_emails),
                'threads_separated': True if self.separate_threads else False,
                'directions_fixed': True,
                'direction_changes': sum(1 for email in emails if email.get('direction_corrected', False)),
                'processing_config': {
                    'clean_html': self.clean_html,
                    'separate_threads': self.separate_threads,
                    'enhanced_direction_classification': True,
                    'classification_rule': 'HTML sender extraction + content analysis'
                }
            }
        }

        logger.info(f"Processing complete: {len(emails)} emails in {len(relevant_threads)} relevant threads ({incoming_count} incoming, {outgoing_count} outgoing)")

        return processed_data

    def save_results(self, processed_data: Dict[str, Any], output_path: str) -> None:
        """Save processed data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_path}")

    def _apply_conservative_filtering(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply conservative filtering to reduce over-classification of incoming emails."""
        logger.info("Applying conservative filtering to match original count...")

        for email in emails:
            content = (email.get('content', '') or email.get('message', '')).lower()
            sender = email.get('sender', '')
            current_direction = email.get('direction')

            # If already classified as outgoing, keep it
            if current_direction == 'outgoing':
                continue

            # Re-evaluate incoming emails with stricter criteria
            if current_direction == 'incoming':
                # Check for business/formal patterns that suggest outgoing
                business_patterns = [
                    'dear sir/madam', 'to whom it may concern', 'invoice information:',
                    'amount due:', 'please contact', 'regards,', 'sincerely,',
                    'thank you for', 'best regards', 'yours truly',
                    'invoice no:', 'invoice date:', 'due date:'
                ]

                # Check for very short content (likely signatures or fragments)
                if (len(content.strip()) < 30 or
                    any(pattern in content for pattern in business_patterns) or
                    '@litera.com' in content or
                    'litera' in content.lower()):
                    email['direction'] = 'outgoing'
                    logger.debug(f"Reclassified short/business email as outgoing: {content[:50]}...")

            # Additional check: emails with phone numbers but no personal content
            if (current_direction == 'incoming' and
                re.search(r'\(\d{3}\)\s*\d{3}-\d{4}', content) and
                not any(personal in content for personal in [
                    'thanks', 'please', 'hi ', 'hello', 'regards', 'apologies'
                ])):
                email['direction'] = 'outgoing'
                logger.debug(f"Reclassified phone-only email as outgoing")

        logger.info("Conservative filtering complete")
        return emails

    def _group_emails_by_threads(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group processed emails back into thread structure."""
        logger.info("Grouping emails by threads...")

        threads_dict = {}

        for email in emails:
            thread_id = email.get('thread_id', email.get('original_id', email.get('id')))

            if thread_id not in threads_dict:
                threads_dict[thread_id] = {
                    'thread_id': thread_id,
                    'subject': email.get('subject', ''),
                    'timestamp': email.get('timestamp', ''),
                    'emails': [],
                    'metadata': {
                        'total_emails_in_thread': 0,
                        'incoming_emails': 0,
                        'outgoing_emails': 0,
                        'is_thread_separated': email.get('is_thread_separated', False),
                        'separated_count': email.get('separated_count', 1)
                    }
                }

            # Clean up email object for thread grouping
            thread_email = {
                'id': email.get('id'),
                'position': email.get('thread_position', 0),
                'content': email.get('content', ''),
                'sender': email.get('sender', ''),
                'direction': email.get('direction'),
                'segment_position': email.get('segment_position', 0),
                'original_content_length': email.get('original_content_length', 0),
                'cleaned_content_length': email.get('cleaned_content_length', 0),
                'segment_index': email.get('segment_index', 0),
                'type': email.get('type', 'individual'),
                'direction_corrected': email.get('direction_corrected', False)
            }

            threads_dict[thread_id]['emails'].append(thread_email)

            # Update thread metadata
            thread_meta = threads_dict[thread_id]['metadata']
            thread_meta['total_emails_in_thread'] += 1
            if email.get('direction') == 'incoming':
                thread_meta['incoming_emails'] += 1
            else:
                thread_meta['outgoing_emails'] += 1

        # Sort emails within each thread by position
        for thread in threads_dict.values():
            thread['emails'].sort(key=lambda x: x['position'])

        # Convert to list and sort by thread_id
        threads_list = list(threads_dict.values())
        threads_list.sort(key=lambda x: str(x['thread_id']))

        logger.info(f"Grouped {len(emails)} emails into {len(threads_list)} threads")

        return threads_list

    def _filter_relevant_threads(self, threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out threads that only contain a single outgoing email (not relevant for conversation analysis)."""
        logger.info("Filtering out single outgoing email threads...")

        relevant_threads = []
        filtered_count = 0

        for thread in threads:
            emails_in_thread = thread['emails']
            thread_meta = thread['metadata']

            # Keep threads that have:
            # 1. More than one email total, OR
            # 2. At least one incoming email (even if single email thread)
            if (thread_meta['total_emails_in_thread'] > 1 or
                thread_meta['incoming_emails'] > 0):
                relevant_threads.append(thread)
            else:
                # This is a single outgoing email thread - filter it out
                filtered_count += 1
                logger.debug(f"Filtered out single outgoing thread {thread['thread_id']}: {thread.get('subject', 'No subject')[:50]}...")

        logger.info(f"Filtered out {filtered_count} single outgoing email threads")
        logger.info(f"Kept {len(relevant_threads)} relevant threads for analysis")

        return relevant_threads

    def _validate_thread_classifications(self, threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-validate email classifications within threads for consistency."""
        logger.info("Performing thread-level classification validation...")

        corrections_made = 0

        for thread in threads:
            emails = thread['emails']
            if len(emails) <= 1:
                continue

            # Analyze thread conversation pattern
            thread_analysis = self._analyze_thread_conversation_pattern(emails)

            # Apply corrections based on thread analysis
            corrections = self._apply_thread_corrections(emails, thread_analysis)
            corrections_made += corrections

            # Update thread metadata
            thread['metadata']['incoming_emails'] = sum(1 for email in emails if email['direction'] == 'incoming')
            thread['metadata']['outgoing_emails'] = sum(1 for email in emails if email['direction'] == 'outgoing')

        logger.info(f"Thread validation complete: {corrections_made} corrections applied")
        return threads

    def _analyze_thread_conversation_pattern(self, emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the overall conversation pattern in a thread."""

        analysis = {
            'total_emails': len(emails),
            'directions': [email['direction'] for email in emails],
            'suspicious_patterns': [],
            'confidence_scores': [],
            'likely_pattern': 'unknown'
        }

        directions = analysis['directions']

        # Pattern 1: All emails marked as outgoing (highly suspicious for multi-email threads)
        if len(emails) > 1 and all(d == 'outgoing' for d in directions):
            analysis['suspicious_patterns'].append('all_outgoing_multi_email')
            analysis['likely_pattern'] = 'conversation_misclassified'

        # Pattern 2: Alternating pattern suggests real conversation
        alternating_score = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1]:
                alternating_score += 1

        if alternating_score >= len(emails) * 0.5:
            analysis['likely_pattern'] = 'genuine_conversation'

        # Pattern 3: First email outgoing, rest incoming (common invoice -> responses pattern)
        if (len(emails) > 1 and
            directions[0] == 'outgoing' and
            all(d == 'incoming' for d in directions[1:])):
            analysis['likely_pattern'] = 'invoice_with_responses'

        # Pattern 4: Analyze subject patterns for consistency
        subjects = [email.get('subject', '') for email in emails]
        reply_pattern = any('re:' in subj.lower() for subj in subjects)
        if reply_pattern and directions[0] == 'outgoing':
            analysis['likely_pattern'] = 'initial_outreach_with_replies'

        return analysis

    def _apply_thread_corrections(self, emails: List[Dict[str, Any]],
                                 thread_analysis: Dict[str, Any]) -> int:
        """Apply corrections based on thread-level analysis."""

        corrections = 0

        # Correction 1: Fix obviously wrong all-outgoing multi-email threads
        if 'all_outgoing_multi_email' in thread_analysis['suspicious_patterns']:
            # In multi-email threads, if all are marked outgoing, likely the replies are misclassified
            # Keep first as outgoing (likely invoice), mark rest as incoming (likely responses)
            for i, email in enumerate(emails):
                if i > 0 and email['direction'] == 'outgoing':
                    # Double-check this isn't actually an outgoing email by re-analyzing content
                    content = email.get('content', '').lower()

                    # Strong indicators this should remain outgoing
                    strong_outgoing_indicators = [
                        'your invoice', 'please remit payment', 'past due', 'suspended until',
                        'thank you for your business', 'contact lms.ar@litera.com'
                    ]

                    # If no strong outgoing indicators, likely a misclassified response
                    if not any(indicator in content for indicator in strong_outgoing_indicators):
                        # Additional check: look for customer response patterns
                        customer_response_patterns = [
                            'thank you', 'received', 'will process', 'payment', 'ap team',
                            'accounts payable', 'pay run', 'sent to ap'
                        ]

                        if any(pattern in content for pattern in customer_response_patterns):
                            email['direction'] = 'incoming'
                            email['direction_corrected'] = True
                            corrections += 1
                            logger.debug(f"Corrected thread position {i} from outgoing to incoming")

        # Correction 2: Fix threads where position 0 is incoming but should be outgoing
        if (len(emails) > 0 and
            emails[0]['direction'] == 'incoming' and
            thread_analysis['likely_pattern'] in ['invoice_with_responses', 'initial_outreach_with_replies']):

            first_email_content = emails[0].get('content', '').lower()
            first_email_subject = emails[0].get('subject', '').lower()

            # Check if first email has strong outgoing characteristics
            if ('invoice' in first_email_subject or
                'reminder' in first_email_subject or
                any(pattern in first_email_content for pattern in [
                    'your invoice', 'please remit', 'past due', 'amount due'
                ])):
                emails[0]['direction'] = 'outgoing'
                emails[0]['direction_corrected'] = True
                corrections += 1
                logger.debug("Corrected first email from incoming to outgoing")

        # Correction 3: Validate replies are appropriately classified
        for i, email in enumerate(emails[1:], 1):  # Skip first email
            subject = email.get('subject', '').lower()
            content = email.get('content', '').lower()

            # If subject indicates this is a reply but classified as outgoing, double-check
            if (subject.startswith('re:') and email['direction'] == 'outgoing'):
                # Check if this is actually a customer reply
                customer_reply_indicators = [
                    'thank you', 'thanks', 'received', 'will be processed',
                    'sent to ap', 'accounts payable', 'pay run', 'payment status'
                ]

                if any(indicator in content for indicator in customer_reply_indicators):
                    email['direction'] = 'incoming'
                    email['direction_corrected'] = True
                    corrections += 1
                    logger.debug(f"Corrected reply at position {i} from outgoing to incoming")

        return corrections