#!/usr/bin/env python3
"""
Taxonomy curation module for email taxonomy pipeline.

Converts LLM analysis results into final curated taxonomy files.
"""

import json
import yaml
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging
import re
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class TaxonomyCurator:
    """Curates LLM analysis results into final taxonomy files."""

    def __init__(self):
        self.curation_rules = {
            # Intent consolidation rules - More granular categories
            'intent_consolidation': {
                'Payment Status Inquiry': ['payment status', 'payment inquiry', 'payment question'],
                'Payment Promise': ['payment promise', 'payment commitment', 'will pay'],
                'Hardship Communication': ['hardship', 'financial difficulty', 'cannot pay'],
                'Dispute Resolution': ['dispute', 'challenge', 'disagree', 'error'],
                'Invoice Documentation': ['invoice request', 'documentation', 'w9', 'receipt'],
                'Account Information Update': ['account update', 'information update', 'contact change'],
                'Payment Method Inquiry': ['payment method', 'payment option', 'how to pay'],
                'Settlement Negotiation': ['settlement', 'payment arrangement', 'negotiate'],
                'Acknowledgment': ['acknowledge', 'received', 'confirm receipt'],
                'Third Party Authorization': ['lawyer', 'attorney', 'representative']
            },
            # Sentiment consolidation rules - More emotional granularity
            'sentiment_consolidation': {
                'Apologetic': ['apologetic', 'sorry', 'regret', 'apologize'],
                'Frustrated': ['frustrated', 'angry', 'upset', 'dissatisfied'],
                'Cooperative': ['cooperative', 'willing', 'helpful', 'collaborative'],
                'Defensive': ['defensive', 'excuse', 'not my fault', 'justify'],
                'Urgent': ['urgent', 'emergency', 'asap', 'immediate'],
                'Professional': ['professional', 'formal', 'business', 'neutral'],
                'Confused': ['confused', 'unclear', 'don\'t understand', 'clarification'],
                'Grateful': ['grateful', 'thank', 'appreciate', 'thankful'],
                'Desperate': ['desperate', 'pleading', 'help', 'severe']
            }
        }

        # Initialize sentence transformer for semantic similarity
        self.similarity_model = None
        self.similarity_threshold = 0.60  # Lower threshold to preserve distinct categories
        self.sentiment_similarity_threshold = 0.50  # Even lower threshold for sentiment to preserve emotional nuance

    def _get_similarity_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self.similarity_model is None:
            logger.info("Loading sentence transformer model for semantic similarity...")
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.similarity_model

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using sentence transformers."""
        model = self._get_similarity_model()
        embeddings = model.encode([text1, text2])

        # Calculate cosine similarity using numpy
        vec1, vec2 = embeddings[0], embeddings[1]
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def _has_distinct_business_value(self, cat1_name: str, cat2_name: str) -> bool:
        """Check if two categories have distinct business value that should be preserved."""
        cat1_lower = cat1_name.lower()
        cat2_lower = cat2_name.lower()

        # Emotional sentiment categories that must remain distinct
        emotional_distinctions = [
            ('apologetic', 'frustrated'),    # Very different emotional states
            ('apologetic', 'defensive'),     # Different emotional responses
            ('frustrated', 'cooperative'),   # Opposite emotional states
            ('frustrated', 'grateful'),      # Opposite emotional states
            ('defensive', 'cooperative'),    # Different customer attitudes
            ('urgent', 'apologetic'),        # Different emotional urgency
            ('desperate', 'professional'),   # Very different emotional states
            ('confused', 'defensive'),       # Different emotional responses
            ('grateful', 'desperate'),       # Different emotional states
        ]

        # Business intent keywords that should remain separate
        intent_distinctions = [
            ('payment', 'invoice'),          # Payment vs Invoice management are different
            ('payment', 'dispute'),          # Payment vs Dispute handling are different
            ('invoice', 'dispute'),          # Invoice vs Dispute are different
            ('inquiry', 'promise'),          # Inquiry vs Promise are different actions
            ('hardship', 'settlement'),      # Hardship vs Settlement are different situations
            ('documentation', 'status'),     # Documentation vs Status requests are different
            ('acknowledgment', 'authorization'), # Different types of communication
        ]

        # Check emotional distinctions first (most important for sentiment)
        for emotion1, emotion2 in emotional_distinctions:
            if ((emotion1 in cat1_lower and emotion2 in cat2_lower) or
                (emotion2 in cat1_lower and emotion1 in cat2_lower)):
                logger.debug(f"Preserving distinct emotional states: '{cat1_name}' vs '{cat2_name}'")
                return True

        # Check business intent distinctions
        for intent1, intent2 in intent_distinctions:
            if ((intent1 in cat1_lower and intent2 in cat2_lower) or
                (intent2 in cat1_lower and intent1 in cat2_lower)):
                logger.debug(f"Preserving distinct business intents: '{cat1_name}' vs '{cat2_name}'")
                return True

        # Special case: "information update" variants can still be merged
        info_update_variants = ['information update', 'account information', 'update notification']
        cat1_has_info_update = any(variant in cat1_lower for variant in info_update_variants)
        cat2_has_info_update = any(variant in cat2_lower for variant in info_update_variants)

        if cat1_has_info_update and cat2_has_info_update:
            logger.debug(f"Both categories are information update variants: '{cat1_name}' and '{cat2_name}' - allowing merge")
            return False  # Allow merging of information update variants

        return False

    def _merge_similar_categories(self, categories: Dict[str, Any], threshold: float = None) -> Dict[str, Any]:
        """Merge categories with high semantic similarity, preserving distinct business value."""
        if threshold is None:
            threshold = self.similarity_threshold

        logger.info(f"Merging similar categories with threshold {threshold}")
        logger.info(f"Input categories: {list(categories.keys())}")

        category_names = list(categories.keys())
        merged_categories = {}
        processed = set()

        for i, category1 in enumerate(category_names):
            if category1 in processed:
                logger.debug(f"Skipping '{category1}' - already processed")
                continue

            logger.info(f"Processing category '{category1}' for potential merges...")

            # Start with this category as the merge target
            merged_name = category1
            merged_data = categories[category1].copy()
            similar_categories = [category1]

            for j, category2 in enumerate(category_names[i+1:], i+1):
                if category2 in processed:
                    logger.debug(f"  Skipping '{category2}' - already processed")
                    continue

                logger.debug(f"  Comparing '{category1}' with '{category2}'")

                # Check if categories have distinct business value
                has_distinct_value = self._has_distinct_business_value(category1, category2)
                logger.debug(f"    Business value check: {has_distinct_value} (True=distinct, False=can merge)")

                if has_distinct_value:
                    logger.debug(f"    SKIPPING: Preserving distinct business value between '{category1}' and '{category2}'")
                    continue

                # Calculate similarity between descriptions
                desc1 = categories[category1].get('definition', '')
                desc2 = categories[category2].get('definition', '')

                similarity = self._calculate_semantic_similarity(desc1, desc2)
                logger.debug(f"    Semantic similarity: {similarity:.4f} (threshold: {threshold})")

                if similarity > threshold:
                    logger.info(f"    ✅ MERGING '{category2}' into '{category1}' (similarity: {similarity:.3f})")
                    similar_categories.append(category2)

                    # Merge data
                    merged_data['clusters'].extend(categories[category2].get('clusters', []))
                    merged_data['total_emails'] += categories[category2].get('total_emails', 0)

                    # Combine indicators and rules
                    if 'sample_indicators' in merged_data and 'sample_indicators' in categories[category2]:
                        merged_data['sample_indicators'].extend(categories[category2]['sample_indicators'])
                    if 'decision_rules' in merged_data and 'decision_rules' in categories[category2]:
                        merged_data['decision_rules'].extend(categories[category2]['decision_rules'])

                    processed.add(category2)
                else:
                    logger.info(f"    ❌ Not merging - similarity {similarity:.4f} below threshold {threshold}")

            # Use the most representative name (could be improved with better logic)
            if len(similar_categories) > 1:
                # Choose the name with highest email count
                best_name = max(similar_categories, key=lambda x: categories[x].get('total_emails', 0))
                merged_name = best_name
                merged_data = categories[best_name].copy()

                # Re-merge all data into the best category
                merged_data['clusters'] = []
                merged_data['total_emails'] = 0
                all_indicators = []
                all_rules = []

                for cat in similar_categories:
                    merged_data['clusters'].extend(categories[cat].get('clusters', []))
                    merged_data['total_emails'] += categories[cat].get('total_emails', 0)
                    all_indicators.extend(categories[cat].get('sample_indicators', []))
                    all_rules.extend(categories[cat].get('decision_rules', []))

                # Deduplicate lists
                merged_data['sample_indicators'] = list(set(all_indicators))
                merged_data['decision_rules'] = list(set(all_rules))

            merged_categories[merged_name] = merged_data
            processed.add(category1)

        logger.info(f"Merged {len(categories)} categories into {len(merged_categories)} categories")
        return merged_categories

    def _apply_business_consolidation_rules(self, categories: Dict[str, Any]) -> Dict[str, Any]:
        """Apply business-specific consolidation rules and patterns."""
        logger.info("Applying business consolidation rules...")

        # Pattern-based consolidation rules - only merge truly duplicate administrative patterns
        consolidation_patterns = {
            'Administrative Update': [
                r'^administrative.*update.*and.*inquiry$',
                r'^administrative.*update.*and.*confirmation$',
                r'^administrative.*update.*request$',
                r'^administrative.*communication$',
                r'^contact.*information.*update$'
            ]
            # Removed broader patterns to preserve payment and invoice distinctions
        }

        # Group categories by patterns
        pattern_groups = {}
        unmatched_categories = {}

        for category_name, category_data in categories.items():
            matched = False
            category_lower = category_name.lower()

            for target_name, patterns in consolidation_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, category_lower):
                        if target_name not in pattern_groups:
                            pattern_groups[target_name] = []
                        pattern_groups[target_name].append((category_name, category_data))
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                unmatched_categories[category_name] = category_data

        # Merge pattern groups
        consolidated_categories = {}

        for target_name, group_categories in pattern_groups.items():
            if len(group_categories) == 1:
                # Only one category in this pattern, keep original name
                orig_name, orig_data = group_categories[0]
                consolidated_categories[orig_name] = orig_data
            else:
                # Multiple categories, merge them
                logger.info(f"Consolidating {len(group_categories)} categories into '{target_name}'")

                merged_data = {
                    'definition': self._create_consolidated_definition(target_name, group_categories),
                    'clusters': [],
                    'total_emails': 0,
                    'sample_indicators': [],
                    'decision_rules': [],
                    'business_relevance': self._get_business_relevance(target_name)
                }

                for category_name, category_data in group_categories:
                    merged_data['clusters'].extend(category_data.get('clusters', []))
                    merged_data['total_emails'] += category_data.get('total_emails', 0)
                    merged_data['sample_indicators'].extend(category_data.get('sample_indicators', []))
                    merged_data['decision_rules'].extend(category_data.get('decision_rules', []))

                # Deduplicate lists
                merged_data['sample_indicators'] = list(set(merged_data['sample_indicators']))[:5]  # Limit to 5
                merged_data['decision_rules'] = list(set(merged_data['decision_rules']))[:5]  # Limit to 5

                consolidated_categories[target_name] = merged_data

        # Add unmatched categories
        consolidated_categories.update(unmatched_categories)

        # Ensure minimum category diversity (at least 3 categories for business value)
        if len(consolidated_categories) < 3:
            logger.warning(f"Only {len(consolidated_categories)} categories after consolidation. Consider lowering similarity threshold.")

        return consolidated_categories

    def _create_consolidated_definition(self, target_name: str, group_categories: List[Tuple[str, Dict]]) -> str:
        """Create a consolidated definition for merged categories."""
        definitions = {
            'Administrative Update': 'Customer communications focused on updating contact information, confirming receipt of documents, and addressing routine administrative matters related to billing and account management.',
            'Payment Processing': 'Customer inquiries and communications related to payment status, payment methods, and payment processing confirmation.',
            'Invoice Management': 'Customer requests for invoice documentation, billing clarifications, and follow-up communications regarding specific invoices or charges.'
        }
        return definitions.get(target_name, f'Consolidated category for {target_name.lower()} related communications.')

    def _get_business_relevance(self, category_name: str) -> str:
        """Get business relevance description for consolidated categories."""
        relevance = {
            'Administrative Update': 'Ensures accurate customer records and smooth administrative processes, reducing billing errors and miscommunications.',
            'Payment Processing': 'Tracks payment-related inquiries to improve payment processing and customer satisfaction.',
            'Invoice Management': 'Facilitates proper invoice handling and reduces billing disputes through clear documentation.'
        }
        return relevance.get(category_name, f'Supports collections operations through effective {category_name.lower()} handling.')

    def consolidate_categories(self, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate LLM-proposed categories using curation rules."""
        proposed_taxonomy = llm_analysis.get('proposed_taxonomy', {})
        intent_categories = proposed_taxonomy.get('intent_categories', {})
        sentiment_categories = proposed_taxonomy.get('sentiment_categories', {})

        # Consolidate intents
        consolidated_intents = {}
        for intent_name, intent_data in intent_categories.items():
            # Find best match in consolidation rules
            consolidated_name = self._find_consolidated_name(
                intent_name,
                self.curation_rules['intent_consolidation']
            )

            if consolidated_name not in consolidated_intents:
                consolidated_intents[consolidated_name] = {
                    'definition': intent_data.get('definition', ''),
                    'clusters': [],
                    'total_emails': 0,
                    'original_names': []
                }

            consolidated_intents[consolidated_name]['clusters'].extend(intent_data.get('clusters', []))
            consolidated_intents[consolidated_name]['total_emails'] += intent_data.get('total_emails', 0)
            consolidated_intents[consolidated_name]['original_names'].append(intent_name)

        # Consolidate sentiments
        consolidated_sentiments = {}
        for sentiment_name, sentiment_data in sentiment_categories.items():
            consolidated_name = self._find_consolidated_name(
                sentiment_name,
                self.curation_rules['sentiment_consolidation']
            )

            if consolidated_name not in consolidated_sentiments:
                consolidated_sentiments[consolidated_name] = {
                    'definition': sentiment_data.get('definition', ''),
                    'clusters': [],
                    'total_emails': 0,
                    'original_names': []
                }

            consolidated_sentiments[consolidated_name]['clusters'].extend(sentiment_data.get('clusters', []))
            consolidated_sentiments[consolidated_name]['total_emails'] += sentiment_data.get('total_emails', 0)
            consolidated_sentiments[consolidated_name]['original_names'].append(sentiment_name)

        return {
            'intent_categories': consolidated_intents,
            'sentiment_categories': consolidated_sentiments
        }

    def _find_consolidated_name(self, original_name: str, consolidation_rules: Dict[str, List[str]]) -> str:
        """Find the consolidated category name for an original name."""
        original_lower = original_name.lower()

        for consolidated_name, variants in consolidation_rules.items():
            if any(variant in original_lower for variant in variants):
                return consolidated_name

        # If no match found, use original name (capitalized)
        return original_name.title()

    def generate_taxonomy_yaml(self, consolidated_taxonomy: Dict[str, Any], analysis_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final taxonomy.yaml structure."""

        intent_categories = consolidated_taxonomy['intent_categories']
        sentiment_categories = consolidated_taxonomy['sentiment_categories']

        taxonomy = {
            'metadata': {
                'version': '1.0',
                'created_by': 'Email Taxonomy Discovery Pipeline',
                'total_emails_analyzed': analysis_summary.get('total_emails_in_analyzed_clusters', 0),
                'coverage_percentage': analysis_summary.get('coverage_percentage', 0),
                'clusters_analyzed': analysis_summary.get('clusters_analyzed', 0),
                'model_used': analysis_summary.get('model_used', 'gpt-4o')
            },
            'intent_categories': {},
            'sentiment_categories': {},
            'classification_rules': {
                'intent_priority': list(intent_categories.keys()),
                'sentiment_priority': list(sentiment_categories.keys()),
                'decision_framework': 'Apply intent first, then determine sentiment within that intent context'
            }
        }

        # Add intent categories
        for intent_name, intent_data in intent_categories.items():
            taxonomy['intent_categories'][intent_name] = {
                'definition': self._refine_definition(intent_data['definition'], intent_name),
                'examples': self._generate_examples(intent_name, 'intent'),
                'decision_rules': self._generate_decision_rules(intent_name, 'intent'),
                'coverage': {
                    'total_emails': intent_data['total_emails'],
                    'clusters': len(intent_data['clusters'])
                }
            }

        # Add sentiment categories
        for sentiment_name, sentiment_data in sentiment_categories.items():
            taxonomy['sentiment_categories'][sentiment_name] = {
                'definition': self._refine_definition(sentiment_data['definition'], sentiment_name),
                'examples': self._generate_examples(sentiment_name, 'sentiment'),
                'decision_rules': self._generate_decision_rules(sentiment_name, 'sentiment'),
                'coverage': {
                    'total_emails': sentiment_data['total_emails'],
                    'clusters': len(sentiment_data['clusters'])
                }
            }

        return taxonomy

    def _refine_definition(self, original_definition: str, category_name: str) -> str:
        """Refine category definitions for clarity and consistency."""

        # Predefined refined definitions
        refined_definitions = {
            'Payment Inquiry': 'Questions about payment status, methods, schedules, or acknowledgment of payments made',
            'Invoice Management': 'Requests for invoice details, corrections, documentation, or billing-related clarifications',
            'Information Request': 'General information requests not specifically related to payments or invoices',
            'Cooperative': 'Willing to work together, shows positive engagement and collaboration',
            'Administrative': 'Neutral, business-focused communication without emotional indicators',
            'Informational': 'Seeking or providing factual information in a straightforward manner',
            'Frustrated': 'Expressing dissatisfaction, impatience, or negative emotions about the situation'
        }

        return refined_definitions.get(category_name, original_definition)

    def _generate_examples(self, category_name: str, category_type: str) -> List[str]:
        """Generate example phrases/indicators for each category."""

        examples = {
            'intent': {
                'Payment Inquiry': [
                    'When will my payment be processed?',
                    'I made a payment last week, has it been received?',
                    'What payment methods do you accept?',
                    'Can I set up a payment plan?'
                ],
                'Invoice Management': [
                    'Can you send me a copy of invoice #12345?',
                    'There seems to be an error on my bill',
                    'I need an updated invoice with correct address',
                    'Please explain these charges on my statement'
                ],
                'Information Request': [
                    'What is my current account balance?',
                    'Who should I contact about this matter?',
                    'Can you explain your company policy?',
                    'I need information about my account status'
                ]
            },
            'sentiment': {
                'Cooperative': [
                    'I want to resolve this matter quickly',
                    'Please let me know how I can help',
                    'Thank you for your assistance',
                    'I appreciate your patience'
                ],
                'Administrative': [
                    'Please update my account information',
                    'I am writing to inform you...',
                    'This is regarding account number...',
                    'Per our previous correspondence'
                ],
                'Informational': [
                    'I need clarification on...',
                    'Could you please explain...',
                    'I am inquiring about...',
                    'What is the status of...'
                ],
                'Frustrated': [
                    'This is unacceptable',
                    'I am very disappointed',
                    'This has been going on too long',
                    'Why has this not been resolved?'
                ]
            }
        }

        return examples.get(category_type, {}).get(category_name, [])

    def _generate_decision_rules(self, category_name: str, category_type: str) -> List[str]:
        """Generate decision rules for each category."""

        rules = {
            'intent': {
                'Payment Inquiry': [
                    'Contains questions about payment status or confirmation',
                    'Mentions payment methods, schedules, or arrangements',
                    'Asks about received payments or payment processing'
                ],
                'Invoice Management': [
                    'Requests invoice copies, corrections, or clarifications',
                    'Mentions billing errors or discrepancies',
                    'Asks for documentation or statement details'
                ],
                'Information Request': [
                    'General questions not related to payments or invoices',
                    'Requests for account information or company policies',
                    'Seeks clarification on procedures or contacts'
                ]
            },
            'sentiment': {
                'Cooperative': [
                    'Uses polite language and collaborative tone',
                    'Expresses willingness to resolve issues',
                    'Shows appreciation or patience'
                ],
                'Administrative': [
                    'Uses formal, business-like language',
                    'Neutral tone without emotional indicators',
                    'Focuses on facts and procedures'
                ],
                'Informational': [
                    'Primarily asks questions or seeks clarification',
                    'Neutral to slightly positive tone',
                    'Fact-seeking rather than emotional'
                ],
                'Frustrated': [
                    'Uses negative language or expresses dissatisfaction',
                    'Shows impatience or urgency',
                    'Contains complaints or criticism'
                ]
            }
        }

        return rules.get(category_type, {}).get(category_name, [])

    def generate_labeling_guide(self, taxonomy: Dict[str, Any]) -> str:
        """Generate comprehensive labeling guide markdown."""

        guide = f"""# Customer Email Taxonomy Labeling Guide

Generated by Email Taxonomy Discovery Pipeline

## Overview

This guide provides detailed instructions for classifying INCOMING CUSTOMER emails into intent and sentiment categories. The taxonomy was derived from analysis of {taxonomy['metadata']['total_emails_analyzed']} incoming customer emails with {taxonomy['metadata']['coverage_percentage']:.1f}% coverage.

**Purpose**: Categorize emails RECEIVED by collections teams FROM customers to understand customer communication patterns and improve response strategies.

## Classification Framework

{taxonomy['classification_rules']['decision_framework']}

### Priority Order
- **Intent Categories**: {', '.join(taxonomy['classification_rules']['intent_priority'])}
- **Sentiment Categories**: {', '.join(taxonomy['classification_rules']['sentiment_priority'])}

## Intent Categories

"""

        # Add intent categories
        for intent_name, intent_data in taxonomy['intent_categories'].items():
            guide += f"""### {intent_name}

**Definition**: {intent_data.get('description', intent_data.get('definition', ''))}

**Coverage**: {intent_data.get('coverage', 'N/A')} of emails

**Decision Rules**:
"""
            decision_rules = intent_data.get('decision_rules', [])
            for rule in decision_rules:
                guide += f"- {rule}\n"

            guide += "\n**Examples**:\n"
            examples = intent_data.get('examples', [])
            for example in examples:
                guide += f"- \"{example}\"\n"

            guide += "\n"

        guide += "\n## Sentiment Categories\n\n"

        # Add sentiment categories
        for sentiment_name, sentiment_data in taxonomy['sentiment_categories'].items():
            guide += f"""### {sentiment_name}

**Definition**: {sentiment_data.get('description', sentiment_data.get('definition', ''))}

**Coverage**: {sentiment_data.get('coverage', 'N/A')} of emails

**Decision Rules**:
"""
            decision_rules = sentiment_data.get('decision_rules', [])
            for rule in decision_rules:
                guide += f"- {rule}\n"

            guide += "\n**Examples**:\n"
            examples = sentiment_data.get('examples', [])
            for example in examples:
                guide += f"- \"{example}\"\n"

            guide += "\n"

        guide += """
## Classification Process

1. **Read the complete email** including subject and content
2. **Identify the primary intent** using the decision rules above
3. **Determine the sentiment** within the context of that intent
4. **Apply the most specific category** that fits the email's main purpose
5. **When in doubt**, choose the broader category

## Quality Guidelines

- **Consistency**: Apply the same criteria across all emails
- **Primary Purpose**: Focus on the main intent, not secondary mentions
- **Context Matters**: Consider the overall tone and situation
- **Document Edge Cases**: Note emails that don't fit clearly for future refinement

---

*This guide was generated from analysis of real customer collection emails using GPT-4o and human curation.*
"""

        return guide

    def curate_taxonomy(self, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform complete taxonomy curation from LLM analysis."""
        logger.info("Starting taxonomy curation...")

        # Extract rich content from LLM analysis
        rich_taxonomy = self._extract_rich_taxonomy(llm_analysis)

        # Generate final taxonomy using rich content
        analysis_summary = llm_analysis.get('analysis_summary', {})
        final_taxonomy = self.generate_rich_taxonomy_yaml(rich_taxonomy, analysis_summary)

        # Generate labeling guide using structured data
        structured_taxonomy = self._create_structured_taxonomy_for_guide(rich_taxonomy, analysis_summary)
        labeling_guide = self.generate_labeling_guide(structured_taxonomy)

        results = {
            'final_taxonomy': final_taxonomy,
            'labeling_guide': labeling_guide,
            'curation_stats': {
                'original_intent_categories': len(llm_analysis.get('cluster_analyses', {})),
                'final_intent_categories': len(rich_taxonomy['intent_categories']),
                'original_sentiment_categories': len(set([analysis.get('proposed_sentiment', '').split('/')[0] for analysis in llm_analysis.get('cluster_analyses', {}).values()])),
                'final_sentiment_categories': len(rich_taxonomy['sentiment_categories']),
                'total_emails_covered': analysis_summary.get('total_emails_in_analyzed_clusters', 0),
                'coverage_percentage': analysis_summary.get('coverage_percentage', 0)
            }
        }

        logger.info(f"Curation complete: {len(rich_taxonomy['intent_categories'])} intent + {len(rich_taxonomy['sentiment_categories'])} sentiment categories")

        return results

    def save_results(self, curation_results: Dict[str, Any], output_dir: Path) -> None:
        """Save curated taxonomy files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save taxonomy.yaml (now a formatted string)
        taxonomy_path = output_dir / 'taxonomy.yaml'
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            if isinstance(curation_results['final_taxonomy'], str):
                f.write(curation_results['final_taxonomy'])
            else:
                yaml.dump(curation_results['final_taxonomy'], f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved taxonomy to {taxonomy_path}")

        # Save labeling guide
        guide_path = output_dir / 'taxonomy_labeling_guide.md'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(curation_results['labeling_guide'])
        logger.info(f"Saved labeling guide to {guide_path}")

        # Save curation summary
        summary_path = output_dir / 'curation_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(curation_results['curation_stats'], f, indent=2)
        logger.info(f"Saved curation summary to {summary_path}")

    def _extract_rich_taxonomy(self, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and consolidate rich taxonomy content from cluster analyses."""
        cluster_analyses = llm_analysis.get('cluster_analyses', {})

        # First pass: Extract raw categories from clusters
        intent_categories = {}
        sentiment_categories = {}

        for cluster_id, analysis in cluster_analyses.items():
            # Skip failed analyses that contain errors
            if 'error' in analysis:
                logger.warning(f"Skipping cluster {cluster_id} due to analysis error: {analysis.get('error', 'Unknown error')}")
                continue

            # Skip analyses that don't have required fields
            if not analysis.get('proposed_intent') or not analysis.get('proposed_sentiment'):
                logger.warning(f"Skipping cluster {cluster_id} due to missing intent or sentiment")
                continue

            # Extract intent information
            intent_name = analysis.get('proposed_intent')
            if intent_name not in intent_categories:
                intent_categories[intent_name] = {
                    'definition': analysis.get('intent_definition', ''),
                    'decision_rules': analysis.get('decision_rules', []),
                    'sample_indicators': analysis.get('sample_indicators', []),
                    'business_relevance': analysis.get('business_relevance', ''),
                    'clusters': [],
                    'total_emails': 0
                }

            intent_categories[intent_name]['clusters'].append(cluster_id)
            intent_categories[intent_name]['total_emails'] += analysis.get('cluster_size', 0)

            # Extract sentiment information with emotional markers
            sentiment_name = analysis.get('proposed_sentiment').split('/')[0]
            if sentiment_name not in sentiment_categories:
                sentiment_categories[sentiment_name] = {
                    'definition': analysis.get('sentiment_definition', ''),
                    'emotional_markers': [],
                    'sample_indicators': analysis.get('sample_indicators', []),
                    'clusters': [],
                    'total_emails': 0
                }

            # Add emotional markers for more granular sentiment analysis
            emotional_markers = analysis.get('emotional_markers', [])
            if emotional_markers:
                sentiment_categories[sentiment_name]['emotional_markers'].extend(emotional_markers)

            sentiment_categories[sentiment_name]['clusters'].append(cluster_id)
            sentiment_categories[sentiment_name]['total_emails'] += analysis.get('cluster_size', 0)

        logger.info(f"Extracted {len(intent_categories)} raw intent categories")
        logger.info(f"Extracted {len(sentiment_categories)} raw sentiment categories")

        # DISABLE ALL CONSOLIDATION - Keep ALL categories as-is
        # This preserves every single intent and sentiment category found by the LLM
        # No merging, no filtering, no consolidation

        logger.info("CONSOLIDATION DISABLED: Preserving ALL categories without merging or filtering")
        logger.info(f"Keeping all {len(intent_categories)} intent categories")
        logger.info(f"Keeping all {len(sentiment_categories)} sentiment categories")

        return {
            'intent_categories': intent_categories,  # Return raw categories without consolidation
            'sentiment_categories': sentiment_categories  # Return raw categories without consolidation
        }

    def _generate_realistic_examples(self, intent_name: str, indicators: List[str]) -> List[str]:
        """Generate realistic examples based on indicators."""
        if not indicators:
            return []

        # Create examples based on indicators
        examples = []
        for indicator in indicators[:3]:  # Limit to 3 examples
            if 'update' in indicator.lower():
                examples.append(f"Please {indicator} for our billing records")
            elif 'request' in indicator.lower():
                examples.append(f"I need to {indicator} regarding my account")
            else:
                examples.append(f"Email content containing: {indicator}")

        return examples

    def _get_business_value_for_sentiment(self, sentiment_name: str) -> str:
        """Get business value description for sentiment."""
        business_values = {
            'Cooperative': 'Identify engaged customers likely to resolve issues quickly',
            'Administrative': 'Standard processing - routine business communications',
            'Informational': 'Track information gaps and communication needs',
            'Frustrated': 'Flag for priority handling and relationship management'
        }
        return business_values.get(sentiment_name, 'Track customer sentiment patterns')

    def _generate_sentiment_decision_rules(self, sentiment_name: str) -> List[str]:
        """Generate decision rules for sentiment categories."""
        rules = {
            'Cooperative': [
                'Customer offers to provide additional information',
                'Customer confirms payment actions or timeline',
                'Customer apologizes for delays or issues',
                'Customer actively works toward resolution'
            ],
            'Administrative': [
                'Email reports system or processing issues',
                'Email requests cancellations or corrections due to errors',
                'Email provides administrative status updates',
                'Email mentions processing delays or technical problems'
            ],
            'Informational': [
                'Email confirms successful payment completion',
                'Email provides routine business updates or quotes',
                'Email requests feedback or procedural information',
                'Email is not directly collection-related'
            ],
            'Frustrated': [
                'Customer expresses dissatisfaction or frustration',
                'Customer uses urgent or demanding language',
                'Customer mentions escalation or complaints'
            ]
        }
        return rules.get(sentiment_name, [])

    def _generate_sentiment_indicators(self, sentiment_name: str) -> List[str]:
        """Generate key indicators for sentiment categories."""
        indicators = {
            'Cooperative': [
                'happy to provide',
                'apologies for the delays',
                'trying to resolve this',
                'let me know if you need',
                'payment has been initiated'
            ],
            'Administrative': [
                'cancel the invoice',
                'awaiting payment',
                'processed and currently awaiting',
                'unexpected error',
                'system issue'
            ],
            'Informational': [
                'invoice has been paid',
                'please submit through',
                'draft quote attached',
                'opinion matters',
                'feedback request'
            ],
            'Frustrated': [
                'unacceptable delay',
                'need immediate resolution',
                'escalating this issue',
                'extremely disappointed'
            ]
        }
        return indicators.get(sentiment_name, [])

    def _generate_sentiment_examples(self, sentiment_name: str) -> List[str]:
        """Generate examples for sentiment categories."""
        examples = {
            'Cooperative': [
                'Apologies for the delay, payment will be processed tomorrow',
                'Happy to provide the W9 form you requested',
                'Let me loop in our AP team to resolve this quickly'
            ],
            'Administrative': [
                'Please cancel invoice #INV123 due to system error',
                'Invoice processed and currently awaiting final approval',
                'Unexpected error occurred during payment processing'
            ],
            'Informational': [
                'Your invoice has been paid via check #12345',
                'Please submit future invoices through our portal',
                'Please find attached draft quote as requested'
            ],
            'Frustrated': [
                'This is the third time I have requested this information',
                'Need immediate resolution - this delay is unacceptable'
            ]
        }
        return examples.get(sentiment_name, [])

    def generate_rich_taxonomy_yaml(self, rich_taxonomy: Dict[str, Any], analysis_summary: Dict[str, Any]) -> str:
        """Generate clean, well-formatted taxonomy YAML matching reference structure."""
        intent_categories = rich_taxonomy['intent_categories']
        sentiment_categories = rich_taxonomy['sentiment_categories']

        # Generate the YAML content as a formatted string
        yaml_content = self._generate_formatted_yaml(
            intent_categories,
            sentiment_categories,
            analysis_summary
        )

        return yaml_content

    def _generate_formatted_yaml(self, intent_categories: Dict[str, Any], sentiment_categories: Dict[str, Any], analysis_summary: Dict[str, Any]) -> str:
        """Generate clean, formatted YAML string matching reference taxonomy structure."""

        total_emails = analysis_summary.get('total_emails_in_analyzed_clusters', 0)
        clusters_analyzed = analysis_summary.get('clusters_analyzed', 0)
        coverage_pct = analysis_summary.get('coverage_percentage', 0)

        # Start with header and metadata
        yaml_lines = [
            "# Collection Notes AI - Sentiment and Intent Taxonomy",
            f"# Derived from clustering analysis of {total_emails} real collection emails",
            "# Production-ready categories for NetSuite Collection Notes automation",
            "",
            'version: "1.0"',
            'generated_date: "2024-09-19"',
            f'source_emails: {total_emails}',
            f'clusters_analyzed: {clusters_analyzed}',
            f'coverage: "{coverage_pct:.1f}%"  # {total_emails} emails from top {clusters_analyzed} clusters',
            ""
        ]

        # Add intent categories section
        yaml_lines.extend([
            "# INTENT CATEGORIES",
            f"# Consolidated from {clusters_analyzed} clusters into {len(intent_categories)} actionable categories",
            "intent_categories:",
            ""
        ])

        # Add each intent category
        for intent_name, intent_data in intent_categories.items():
            snake_case_name = self._to_snake_case(intent_name)
            coverage_pct = (intent_data['total_emails'] / total_emails) * 100

            yaml_lines.extend([
                f"  {snake_case_name}:",
                f'    display_name: "{intent_name}"',
                f'    description: "{intent_data["definition"]}"',
                f'    business_value: "{intent_data.get("business_relevance", "Track customer engagement patterns")}"',
                f'    coverage: "{coverage_pct:.1f}%"  # {intent_data["total_emails"]} emails',
                "    decision_rules:"
            ])

            for rule in intent_data.get('decision_rules', []):
                yaml_lines.append(f'      - "{rule}"')

            yaml_lines.append("    key_indicators:")
            for indicator in intent_data.get('sample_indicators', []):
                yaml_lines.append(f'      - "{indicator}"')

            yaml_lines.append("    examples:")
            examples = self._generate_clean_examples(intent_name, intent_data.get('sample_indicators', []))
            for example in examples:
                yaml_lines.append(f'      - "{example}"')

            yaml_lines.append("")

        # Add sentiment categories section
        yaml_lines.extend([
            "# SENTIMENT CATEGORIES",
            f"# Consolidated from {len(sentiment_categories)} categories into distinct sentiment levels",
            "sentiment_categories:",
            ""
        ])

        # Add each sentiment category
        for sentiment_name, sentiment_data in sentiment_categories.items():
            snake_case_name = self._to_snake_case(sentiment_name)
            coverage_pct = (sentiment_data['total_emails'] / total_emails) * 100

            yaml_lines.extend([
                f"  {snake_case_name}:",
                f'    display_name: "{sentiment_name}"',
                f'    description: "{sentiment_data["definition"]}"',
                f'    business_value: "{self._get_business_value_for_sentiment(sentiment_name)}"',
                f'    coverage: "{coverage_pct:.1f}%"  # {sentiment_data["total_emails"]} emails',
                "    decision_rules:"
            ])

            for rule in self._generate_sentiment_decision_rules(sentiment_name):
                yaml_lines.append(f'      - "{rule}"')

            yaml_lines.append("    key_indicators:")
            for indicator in self._generate_sentiment_indicators(sentiment_name):
                yaml_lines.append(f'      - "{indicator}"')

            yaml_lines.append("    examples:")
            for example in self._generate_sentiment_examples(sentiment_name):
                yaml_lines.append(f'      - "{example}"')

            yaml_lines.append("")

        # Add modifier flags section
        yaml_lines.extend([
            "# MODIFIER FLAGS",
            "# Additional flags for special handling",
            "modifier_flags:",
            "",
            "  urgency:",
            '    description: "Communication indicates time-sensitive matter"',
            '    indicators: ["urgent", "asap", "immediately", "deadline", "overdue"]',
            "",
            "  escalation:",
            '    description: "Customer mentions involving management or external parties"',
            '    indicators: ["manager", "supervisor", "escalate", "complaint", "legal"]',
            "",
            "  payment_commitment:",
            '    description: "Customer makes specific payment promise or timeline"',
            '    indicators: ["will pay", "payment scheduled", "check sent", "processing payment"]',
            ""
        ])

        # Add validation rules section
        yaml_lines.extend([
            "# VALIDATION RULES",
            "validation_rules:",
            "  mutual_exclusivity:",
            '    - "Each email must have exactly one intent category"',
            '    - "Each email must have exactly one sentiment category"',
            '    - "Modifier flags are optional and non-exclusive"',
            "",
            "  confidence_thresholds:",
            '    high_confidence: "Clear indicators present, unambiguous categorization"',
            '    medium_confidence: "Some indicators present, minor ambiguity"',
            '    low_confidence: "Weak indicators, significant ambiguity - flag for human review"',
            ""
        ])

        # Add NetSuite integration section
        yaml_lines.extend([
            "# BUSINESS MAPPING",
            "netsuite_integration:",
            "  collection_note_fields:",
            '    sentiment: "Maps to Collection Note Sentiment picklist"',
            '    intent: "Maps to Collection Note Category field"',
            '    modifiers: "Maps to Collection Note Priority/Flags fields"',
            "",
            "  automation_triggers:",
            '    cooperative_payment_inquiry: "Auto-acknowledge receipt, provide status update"',
            '    invoice_management: "Route to billing team, auto-request documentation"',
            '    frustrated: "Priority queue, assign to senior collector"'
        ])

        return "\n".join(yaml_lines)

    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        import re
        # Replace spaces and special chars with underscores, then lowercase
        snake = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip())
        snake = re.sub(r'_+', '_', snake)  # Replace multiple underscores with single
        return snake.strip('_').lower()

    def _generate_clean_examples(self, intent_name: str, indicators: List[str]) -> List[str]:
        """Generate clean, realistic examples for intent categories."""
        if not indicators:
            return [
                f"Customer email requesting {intent_name.lower()}",
                f"Communication regarding {intent_name.lower()}",
                f"Email about {intent_name.lower()}"
            ]

        examples = []
        for i, indicator in enumerate(indicators[:3]):  # Limit to 3 examples
            if 'update' in indicator.lower():
                examples.append(f"Can you {indicator.lower()} for my account?")
            elif 'request' in indicator.lower():
                examples.append(f"I need to {indicator.lower()} regarding this matter")
            elif 'payment' in indicator.lower():
                examples.append(f"Regarding {indicator.lower()}, please advise")
            else:
                examples.append(f"Email mentions: {indicator}")

        return examples[:3] if examples else [
            f"Customer inquiry about {intent_name.lower()}",
            f"Request for {intent_name.lower()}",
            f"Communication regarding {intent_name.lower()}"
        ]

    def _create_structured_taxonomy_for_guide(self, rich_taxonomy: Dict[str, Any], analysis_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured taxonomy dict for labeling guide generation."""
        intent_categories = rich_taxonomy['intent_categories']
        sentiment_categories = rich_taxonomy['sentiment_categories']
        total_emails = analysis_summary.get('total_emails_in_analyzed_clusters', 0)

        structured = {
            'metadata': {
                'total_emails_analyzed': total_emails,
                'coverage_percentage': analysis_summary.get('coverage_percentage', 0)
            },
            'classification_rules': {
                'intent_priority': list(intent_categories.keys()),
                'sentiment_priority': list(sentiment_categories.keys()),
                'decision_framework': 'Apply intent first, then determine sentiment within that intent context'
            },
            'intent_categories': {},
            'sentiment_categories': {}
        }

        # Add intent categories
        for intent_name, intent_data in intent_categories.items():
            coverage_pct = f"{(intent_data['total_emails'] / total_emails) * 100:.1f}%"

            structured['intent_categories'][intent_name] = {
                'description': intent_data['definition'],
                'coverage': coverage_pct,
                'decision_rules': intent_data.get('decision_rules', []),
                'examples': self._generate_clean_examples(intent_name, intent_data.get('sample_indicators', []))
            }

        # Add sentiment categories
        for sentiment_name, sentiment_data in sentiment_categories.items():
            coverage_pct = f"{(sentiment_data['total_emails'] / total_emails) * 100:.1f}%"

            structured['sentiment_categories'][sentiment_name] = {
                'description': sentiment_data['definition'],
                'coverage': coverage_pct,
                'decision_rules': self._generate_sentiment_decision_rules(sentiment_name),
                'examples': self._generate_sentiment_examples(sentiment_name)
            }

        return structured