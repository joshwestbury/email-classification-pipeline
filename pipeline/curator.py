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
import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ConsolidatedCategory(BaseModel):
    """Pydantic model for consolidated taxonomy category."""
    name: str = Field(..., min_length=1, max_length=100, description="Clear, specific category name")
    definition: str = Field(..., min_length=20, max_length=500, description="Precise definition of the category")
    business_value: str = Field(..., min_length=10, max_length=300, description="Business value for collections operations")
    merged_categories: List[str] = Field(..., min_items=1, description="List of original categories merged into this one")
    decision_rules: List[str] = Field(..., min_items=1, max_items=5, description="Clear decision rules for classification")
    key_indicators: List[str] = Field(..., min_items=1, max_items=5, description="UNIQUE phrases that distinguish THIS category from others - must be specific, not generic")


class ConsolidatedTaxonomy(BaseModel):
    """Pydantic model for consolidated taxonomy response."""
    intent_categories: List[ConsolidatedCategory] = Field(..., min_items=3, max_items=5, description="EXACTLY 3-5 distinct intent categories")
    sentiment_categories: List[ConsolidatedCategory] = Field(..., min_items=3, max_items=4, description="EXACTLY 3-4 distinct sentiment categories")
    consolidation_rationale: str = Field(..., min_length=100, max_length=1000, description="Explanation of consolidation decisions")


class TaxonomyCurator:
    """Curates LLM analysis results into final taxonomy files."""

    def __init__(self):
        # Initialize OpenAI client for LLM-based consolidation
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found. LLM-based consolidation will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

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

    def _convert_consolidated_to_rich_format(self, consolidated_taxonomy_obj, original_rich_taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """Convert consolidated taxonomy object back to rich taxonomy format."""

        # Create new rich taxonomy structure
        consolidated_rich_taxonomy = {
            'intent_categories': {},
            'sentiment_categories': {}
        }

        # Convert consolidated intent categories
        for intent_cat in consolidated_taxonomy_obj.intent_categories:
            consolidated_rich_taxonomy['intent_categories'][intent_cat.name] = {
                'definition': intent_cat.definition,
                'business_value': intent_cat.business_value,
                'decision_rules': intent_cat.decision_rules,
                'key_indicators': intent_cat.key_indicators,
                'merged_categories': intent_cat.merged_categories,
                'total_emails': self._sum_emails_from_merged_categories(
                    intent_cat.merged_categories,
                    original_rich_taxonomy['intent_categories']
                ),
                'clusters': self._collect_clusters_from_merged_categories(
                    intent_cat.merged_categories,
                    original_rich_taxonomy['intent_categories']
                ),
                'real_email_examples': self._collect_real_examples_from_merged_categories(
                    intent_cat.merged_categories,
                    original_rich_taxonomy['intent_categories']
                )
            }

        # Convert consolidated sentiment categories
        for sentiment_cat in consolidated_taxonomy_obj.sentiment_categories:
            consolidated_rich_taxonomy['sentiment_categories'][sentiment_cat.name] = {
                'definition': sentiment_cat.definition,
                'business_value': sentiment_cat.business_value,
                'decision_rules': sentiment_cat.decision_rules,
                'key_indicators': sentiment_cat.key_indicators,
                'merged_categories': sentiment_cat.merged_categories,
                'total_emails': self._sum_emails_from_merged_categories(
                    sentiment_cat.merged_categories,
                    original_rich_taxonomy['sentiment_categories']
                ),
                'clusters': self._collect_clusters_from_merged_categories(
                    sentiment_cat.merged_categories,
                    original_rich_taxonomy['sentiment_categories']
                ),
                'real_email_examples': self._collect_real_examples_from_merged_categories(
                    sentiment_cat.merged_categories,
                    original_rich_taxonomy['sentiment_categories']
                )
            }

        return consolidated_rich_taxonomy

    def _sum_emails_from_merged_categories(self, merged_categories: list, original_categories: dict) -> int:
        """Sum email counts from merged categories."""
        total = 0
        for category_name in merged_categories:
            if category_name in original_categories:
                total += original_categories[category_name].get('total_emails', 0)
        return total

    def _collect_clusters_from_merged_categories(self, merged_categories: list, original_categories: dict) -> list:
        """Collect cluster IDs from merged categories."""
        clusters = []
        for category_name in merged_categories:
            if category_name in original_categories:
                clusters.extend(original_categories[category_name].get('clusters', []))
        return list(set(clusters))  # Remove duplicates

    def _collect_real_examples_from_merged_categories(self, merged_categories: list, original_categories: dict) -> list:
        """Collect real email examples from merged categories."""
        all_examples = []
        seen_content_hashes = set()

        for category_name in merged_categories:
            if category_name in original_categories:
                examples = original_categories[category_name].get('real_email_examples', [])
                for example in examples:
                    # Deduplicate using content hash
                    content_hash = hash(example.get('content', '')[:200])
                    if content_hash not in seen_content_hashes:
                        all_examples.append(example)
                        seen_content_hashes.add(content_hash)

                    # Limit to 3 most diverse examples
                    if len(all_examples) >= 3:
                        return all_examples

        return all_examples

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

        # Add intent categories - use LLM-generated content directly
        for intent_name, intent_data in intent_categories.items():
            taxonomy['intent_categories'][intent_name] = {
                'definition': intent_data.get('definition', ''),
                'examples': intent_data.get('sample_indicators', []),
                'decision_rules': intent_data.get('decision_rules', []),
                'coverage': {
                    'total_emails': intent_data['total_emails'],
                    'clusters': len(intent_data['clusters'])
                }
            }

        # Add sentiment categories - use LLM-generated content directly
        for sentiment_name, sentiment_data in sentiment_categories.items():
            taxonomy['sentiment_categories'][sentiment_name] = {
                'definition': sentiment_data.get('definition', ''),
                'examples': sentiment_data.get('sample_indicators', []),
                'decision_rules': sentiment_data.get('decision_rules', []),
                'coverage': {
                    'total_emails': sentiment_data['total_emails'],
                    'clusters': len(sentiment_data['clusters'])
                }
            }

        return taxonomy

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

    def curate_taxonomy(self, llm_analysis: Dict[str, Any], email_data: Dict[str, Any] = None, cluster_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform complete taxonomy curation from LLM analysis.

        Args:
            llm_analysis: LLM analysis results with cluster information
            email_data: Processed or anonymized email data (optional, for real examples)
            cluster_data: Cluster results with email-to-cluster mappings (optional, for real examples)
        """
        logger.info("Starting taxonomy curation...")

        # Extract rich content from LLM analysis
        rich_taxonomy = self._extract_rich_taxonomy(llm_analysis)

        # Extract real email examples if data is available
        if email_data and cluster_data:
            logger.info("Extracting real email examples from clusters...")
            rich_taxonomy = self._add_real_email_examples(rich_taxonomy, email_data, cluster_data)
        else:
            logger.warning("Email/cluster data not provided - using synthetic examples")

        # Apply LLM-based consolidation to reduce granular categories
        if self.client:
            logger.info("Applying LLM consolidation to reduce category count...")
            try:
                consolidated_taxonomy_obj = self._llm_consolidate_taxonomy(rich_taxonomy)

                # Validate indicator uniqueness and store results
                duplicates = self._validate_indicator_uniqueness(consolidated_taxonomy_obj)

                # Convert consolidated taxonomy object back to rich taxonomy format
                consolidated_rich_taxonomy = self._convert_consolidated_to_rich_format(consolidated_taxonomy_obj, rich_taxonomy)

                # Store indicator uniqueness issues in the taxonomy for reporting
                consolidated_rich_taxonomy['indicator_uniqueness_issues'] = duplicates

                logger.info(f"Consolidation successful: {len(rich_taxonomy['intent_categories'])} → {len(consolidated_rich_taxonomy['intent_categories'])} intents, "
                           f"{len(rich_taxonomy['sentiment_categories'])} → {len(consolidated_rich_taxonomy['sentiment_categories'])} sentiments")
                rich_taxonomy = consolidated_rich_taxonomy

            except Exception as e:
                logger.warning(f"LLM consolidation failed: {e}. Proceeding with original taxonomy.")
        else:
            logger.warning("OpenAI client not available. Skipping LLM consolidation.")

        # Generate final taxonomy using consolidated rich content
        analysis_summary = llm_analysis.get('analysis_summary', {})
        final_taxonomy = self.generate_rich_taxonomy_yaml(rich_taxonomy, analysis_summary)

        # Generate labeling guide using structured data
        structured_taxonomy = self._create_structured_taxonomy_for_guide(rich_taxonomy, analysis_summary)
        labeling_guide = self.generate_labeling_guide(structured_taxonomy)

        # Generate system prompt for production use
        system_prompt = self.generate_system_prompt(rich_taxonomy)

        # Check for any quality issues with indicators
        indicator_uniqueness_issues = rich_taxonomy.get('indicator_uniqueness_issues', {})

        results = {
            'final_taxonomy': final_taxonomy,
            'labeling_guide': labeling_guide,
            'system_prompt': system_prompt,
            'curation_stats': {
                'original_intent_categories': len(llm_analysis.get('cluster_analyses', {})),
                'final_intent_categories': len(rich_taxonomy['intent_categories']),
                'original_sentiment_categories': len(set([analysis.get('proposed_sentiment', '').split('/')[0] for analysis in llm_analysis.get('cluster_analyses', {}).values()])),
                'final_sentiment_categories': len(rich_taxonomy['sentiment_categories']),
                'total_emails_covered': analysis_summary.get('total_emails_in_analyzed_clusters', 0),
                'coverage_percentage': analysis_summary.get('coverage_percentage', 0),
                'indicator_uniqueness_issues': len(indicator_uniqueness_issues),
                'duplicate_indicators': indicator_uniqueness_issues if indicator_uniqueness_issues else None
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

        # Save system prompt for production use
        if 'system_prompt' in curation_results:
            prompt_path = output_dir / 'system_prompt.txt'
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(curation_results['system_prompt'])
            logger.info(f"Saved production system prompt to {prompt_path}")

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
                    'decision_rules': [],  # Add decision_rules field
                    'emotional_markers': [],
                    'sample_indicators': analysis.get('sample_indicators', []),
                    'business_relevance': analysis.get('business_relevance', ''),  # Add business_relevance
                    'clusters': [],
                    'total_emails': 0
                }

            # Add emotional markers for more granular sentiment analysis
            emotional_markers = analysis.get('emotional_markers', [])
            if emotional_markers:
                sentiment_categories[sentiment_name]['emotional_markers'].extend(emotional_markers)

            # Extract sentiment-specific decision rules from the LLM analysis
            # The LLM provides decision_rules that cover both intent and sentiment
            all_decision_rules = analysis.get('decision_rules', [])
            sentiment_rules = [rule for rule in all_decision_rules if 'sentiment' in rule.lower() or sentiment_name.lower() in rule.lower()]
            if sentiment_rules:
                for rule in sentiment_rules:
                    if rule not in sentiment_categories[sentiment_name]['decision_rules']:
                        sentiment_categories[sentiment_name]['decision_rules'].append(rule)

            sentiment_categories[sentiment_name]['clusters'].append(cluster_id)
            sentiment_categories[sentiment_name]['total_emails'] += analysis.get('cluster_size', 0)

        logger.info(f"Extracted {len(intent_categories)} raw intent categories")
        logger.info(f"Extracted {len(sentiment_categories)} raw sentiment categories")

        # Use LLM-based consolidation to intelligently merge similar categories
        granular_taxonomy = {
            'intent_categories': intent_categories,
            'sentiment_categories': sentiment_categories
        }

        if self.client and (len(intent_categories) > 5 or len(sentiment_categories) > 4):
            logger.info("Taxonomy has too many categories - applying LLM-based consolidation...")
            try:
                consolidated = self._llm_consolidate_taxonomy(granular_taxonomy)

                # Convert consolidated categories back to the expected format
                consolidated_intents = {}
                consolidated_sentiments = {}

                for intent in consolidated.intent_categories:
                    # Calculate total emails from merged categories
                    total_emails = sum(
                        intent_categories[orig_name].get('total_emails', 0)
                        for orig_name in intent.merged_categories
                        if orig_name in intent_categories
                    )

                    # Combine clusters from merged categories
                    all_clusters = []
                    for orig_name in intent.merged_categories:
                        if orig_name in intent_categories:
                            all_clusters.extend(intent_categories[orig_name].get('clusters', []))

                    consolidated_intents[intent.name] = {
                        'definition': intent.definition,
                        'decision_rules': intent.decision_rules,
                        'sample_indicators': intent.key_indicators,
                        'business_relevance': intent.business_value,
                        'clusters': all_clusters,
                        'total_emails': total_emails,
                        'merged_from': intent.merged_categories
                    }

                for sentiment in consolidated.sentiment_categories:
                    # Calculate total emails from merged categories
                    total_emails = sum(
                        sentiment_categories[orig_name].get('total_emails', 0)
                        for orig_name in sentiment.merged_categories
                        if orig_name in sentiment_categories
                    )

                    # Combine clusters from merged categories
                    all_clusters = []
                    for orig_name in sentiment.merged_categories:
                        if orig_name in sentiment_categories:
                            all_clusters.extend(sentiment_categories[orig_name].get('clusters', []))

                    consolidated_sentiments[sentiment.name] = {
                        'definition': sentiment.definition,
                        'decision_rules': sentiment.decision_rules,
                        'sample_indicators': sentiment.key_indicators,
                        'business_relevance': sentiment.business_value,
                        'clusters': all_clusters,
                        'total_emails': total_emails,
                        'merged_from': sentiment.merged_categories
                    }

                logger.info(f"LLM consolidation successful: {len(consolidated_intents)} intents, {len(consolidated_sentiments)} sentiments")
                logger.info(f"Consolidation rationale: {consolidated.consolidation_rationale}")

                # Validate indicator uniqueness
                duplicates = self._validate_indicator_uniqueness(consolidated)

                return {
                    'intent_categories': consolidated_intents,
                    'sentiment_categories': consolidated_sentiments,
                    'consolidation_applied': True,
                    'consolidation_rationale': consolidated.consolidation_rationale,
                    'indicator_uniqueness_issues': duplicates
                }

            except Exception as e:
                logger.warning(f"LLM consolidation failed: {e}. Falling back to granular taxonomy.")
                return granular_taxonomy
        else:
            logger.info("Taxonomy size is reasonable - keeping granular categories")
            return granular_taxonomy

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


    def _add_real_email_examples(self, rich_taxonomy: Dict[str, Any], email_data: Dict[str, Any], cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract real anonymized email examples from clusters for each category.

        Args:
            rich_taxonomy: Taxonomy with category definitions and cluster mappings
            email_data: Processed or anonymized email data with threads
            cluster_data: Cluster results with email-to-cluster mappings

        Returns:
            Rich taxonomy enhanced with real email examples
        """
        logger.info("Extracting real email examples from clusters...")

        # Build email lookup: index -> email content
        email_lookup = {}
        threads = email_data.get('threads', [])

        email_index = 0
        for thread in threads:
            for email in thread.get('emails', []):
                # Only use incoming customer emails
                if email.get('direction') == 'incoming':
                    content = email.get('content', '').strip()
                    if content:
                        email_lookup[email_index] = {
                            'content': content,
                            'subject': thread.get('subject', ''),
                            'email_id': email.get('id', '')
                        }
                email_index += 1

        logger.info(f"Built email lookup with {len(email_lookup)} incoming emails")

        # Get cluster assignments
        cluster_labels = cluster_data.get('cluster_labels', [])

        # Build cluster -> email indices mapping
        cluster_to_emails = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id != -1 and idx in email_lookup:  # Skip noise (-1) and missing emails
                cluster_id_str = str(cluster_id)
                if cluster_id_str not in cluster_to_emails:
                    cluster_to_emails[cluster_id_str] = []
                cluster_to_emails[cluster_id_str].append(idx)

        logger.info(f"Mapped {len(cluster_to_emails)} clusters to email indices")

        # Extract examples for intent categories
        for intent_name, intent_data in rich_taxonomy.get('intent_categories', {}).items():
            clusters = intent_data.get('clusters', [])
            examples = self._extract_examples_from_clusters(
                clusters, cluster_to_emails, email_lookup, num_examples=3
            )
            if examples:
                intent_data['real_email_examples'] = examples
                logger.info(f"Extracted {len(examples)} real examples for intent '{intent_name}'")
            else:
                logger.warning(f"No real examples found for intent '{intent_name}' - will use synthetic")

        # Extract examples for sentiment categories
        for sentiment_name, sentiment_data in rich_taxonomy.get('sentiment_categories', {}).items():
            clusters = sentiment_data.get('clusters', [])
            examples = self._extract_examples_from_clusters(
                clusters, cluster_to_emails, email_lookup, num_examples=3
            )
            if examples:
                sentiment_data['real_email_examples'] = examples
                logger.info(f"Extracted {len(examples)} real examples for sentiment '{sentiment_name}'")
            else:
                logger.warning(f"No real examples found for sentiment '{sentiment_name}' - will use synthetic")

        return rich_taxonomy

    def _extract_examples_from_clusters(self, cluster_ids: List[str], cluster_to_emails: Dict[str, List[int]],
                                       email_lookup: Dict[int, Dict[str, str]], num_examples: int = 3) -> List[Dict[str, str]]:
        """
        Extract diverse, representative email examples from a list of clusters.

        Args:
            cluster_ids: List of cluster IDs to extract examples from
            cluster_to_emails: Mapping of cluster ID to email indices
            email_lookup: Mapping of email index to email data
            num_examples: Number of examples to extract

        Returns:
            List of email examples with subject and content
        """
        examples = []
        seen_content_hashes = set()

        # Try to get examples from different clusters for diversity
        for cluster_id in cluster_ids:
            cluster_id_str = str(cluster_id)
            email_indices = cluster_to_emails.get(cluster_id_str, [])

            if not email_indices:
                continue

            # Try up to 5 emails from this cluster to find diverse examples
            for idx in email_indices[:5]:
                email_data = email_lookup.get(idx)
                if not email_data:
                    continue

                content = email_data['content']

                # Skip if content is too short or too long
                if len(content) < 50 or len(content) > 1000:
                    continue

                # Check for diversity (avoid near-duplicate examples)
                content_hash = hash(content[:200])  # Hash first 200 chars for similarity check
                if content_hash in seen_content_hashes:
                    continue

                # Clean up content for presentation
                clean_content = self._clean_email_for_example(content)

                examples.append({
                    'subject': email_data.get('subject', 'No subject'),
                    'content': clean_content
                })

                seen_content_hashes.add(content_hash)

                if len(examples) >= num_examples:
                    return examples

            if len(examples) >= num_examples:
                break

        return examples

    def _clean_email_for_example(self, content: str) -> str:
        """Clean email content for use as an example in taxonomy/prompts."""
        # Remove excessive whitespace
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Limit to first ~300 characters for conciseness
        cleaned = '\n'.join(lines)
        if len(cleaned) > 400:
            # Truncate and add ellipsis
            cleaned = cleaned[:400].rsplit(' ', 1)[0] + '...'

        return cleaned

    def _validate_indicator_uniqueness(self, consolidated_taxonomy: ConsolidatedTaxonomy) -> Dict[str, List[str]]:
        """
        Validate that key indicators are unique across categories.

        Returns a dict of {indicator: [category_names]} for any indicators that appear in multiple categories.
        """
        all_categories = list(consolidated_taxonomy.intent_categories) + list(consolidated_taxonomy.sentiment_categories)

        # Build indicator -> categories mapping
        indicator_to_categories = {}

        for category in all_categories:
            for indicator in category.key_indicators:
                indicator_lower = indicator.lower().strip()
                if indicator_lower not in indicator_to_categories:
                    indicator_to_categories[indicator_lower] = []
                indicator_to_categories[indicator_lower].append(category.name)

        # Find duplicates
        duplicates = {
            indicator: categories
            for indicator, categories in indicator_to_categories.items()
            if len(categories) > 1
        }

        if duplicates:
            logger.warning(f"Found {len(duplicates)} indicators appearing in multiple categories:")
            for indicator, categories in duplicates.items():
                logger.warning(f"  '{indicator}' appears in: {', '.join(categories)}")

        return duplicates

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _llm_consolidate_taxonomy(self, granular_taxonomy: Dict[str, Any]) -> ConsolidatedTaxonomy:
        """Use LLM to intelligently consolidate granular taxonomy into meaningful categories."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Cannot perform LLM-based consolidation.")

        logger.info("Starting LLM-based taxonomy consolidation...")

        # Prepare the granular taxonomy data for the LLM
        intent_data = []
        sentiment_data = []

        for intent_name, intent_info in granular_taxonomy['intent_categories'].items():
            intent_data.append({
                'name': intent_name,
                'definition': intent_info.get('definition', ''),
                'email_count': intent_info.get('total_emails', 0),
                'key_indicators': intent_info.get('sample_indicators', [])[:3]
            })

        for sentiment_name, sentiment_info in granular_taxonomy['sentiment_categories'].items():
            sentiment_data.append({
                'name': sentiment_name,
                'definition': sentiment_info.get('definition', ''),
                'email_count': sentiment_info.get('total_emails', 0),
                'key_indicators': sentiment_info.get('sample_indicators', [])[:3]
            })

        # Sort by email count to prioritize major categories
        intent_data.sort(key=lambda x: x['email_count'], reverse=True)
        sentiment_data.sort(key=lambda x: x['email_count'], reverse=True)

        prompt = f"""You are a business taxonomy expert specializing in customer communication analysis for collections operations.

Your task is to consolidate a granular taxonomy of customer email categories into a streamlined, actionable taxonomy suitable for business operations.

## CURRENT TAXONOMY TO CONSOLIDATE

**INTENT CATEGORIES ({len(intent_data)} categories):**
{json.dumps(intent_data, indent=2)}

**SENTIMENT CATEGORIES ({len(sentiment_data)} categories):**
{json.dumps(sentiment_data, indent=2)}

## CONSOLIDATION REQUIREMENTS

**Target Output:**
- 3-5 Intent Categories that capture distinct customer purposes
- 3-4 Sentiment Categories that reflect meaningful emotional tones

**Consolidation Principles:**
1. **Semantic Similarity**: Merge categories that have similar meanings or serve similar business purposes
2. **Operational Distinctness**: Keep categories separate ONLY if they require fundamentally different agent responses
3. **Data-Driven Naming**: Use category names that emerge from the actual patterns observed in the data
4. **Eliminate Redundancy**: Merge overlapping or redundant categories while preserving meaningful distinctions

**Intent Consolidation Guidelines:**
- Identify which intent categories describe similar customer purposes
- Merge categories that would trigger the same operational response
- Preserve distinctions that genuinely affect how collections agents should respond
- Choose consolidated category names that best represent the merged group

**Sentiment Consolidation Guidelines:**
- Identify which sentiment categories describe similar emotional tones or communication styles
- Merge categories with overlapping emotional characteristics
- Preserve distinctions that require different handling approaches
- Choose consolidated category names that authentically reflect the emotional patterns observed

**Critical Instructions:**
- Category names should emerge from the DATA, not from preconceptions
- If consolidation results in too many categories, continue merging the most similar ones
- Focus on business utility - each category should provide actionable insights
- Respect the actual communication patterns present in the analyzed emails

**KEY INDICATORS REQUIREMENT - CRITICAL:**
- Each category MUST have UNIQUE key indicators that distinguish it from OTHER categories
- Key indicators must be SPECIFIC phrases or patterns that appear ONLY in this category
- Avoid generic indicators that could apply to multiple categories (e.g., "please update your records")
- Good indicators: "payment arrangement request", "dispute invoice charges", "hardship circumstances"
- Bad indicators: "thank you", "please", "regarding account" (too generic)
- If you find the same indicator appearing in multiple categories, it MUST be removed or made more specific
- Think: "What phrases would let me distinguish THIS category from similar ones?"

## OUTPUT FORMAT

Respond with ONLY a valid JSON object matching this exact structure:

{{
    "intent_categories": [
        {{
            "name": "Clear Category Name",
            "definition": "Precise definition of what customers want to achieve",
            "business_value": "Why this matters for collections operations",
            "merged_categories": ["Original Category 1", "Original Category 2"],
            "decision_rules": ["Rule 1", "Rule 2", "Rule 3"],
            "key_indicators": ["indicator1", "indicator2", "indicator3"]
        }}
    ],
    "sentiment_categories": [
        {{
            "name": "Clear Sentiment Name",
            "definition": "Precise definition of emotional tone/communication style",
            "business_value": "Why this sentiment requires specific handling",
            "merged_categories": ["Original Sentiment 1", "Original Sentiment 2"],
            "decision_rules": ["Rule 1", "Rule 2", "Rule 3"],
            "key_indicators": ["indicator1", "indicator2", "indicator3"]
        }}
    ],
    "consolidation_rationale": "Brief explanation of your consolidation decisions and why the resulting categories are optimal for collections operations"
}}

CRITICAL: Return ONLY the JSON object. No additional text, explanations, or markdown formatting."""

        logger.info(f"Consolidation prompt length: {len(prompt)} characters")
        logger.debug(f"Consolidation prompt (first 1000 chars): {prompt[:1000]}...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a business taxonomy expert. Respond with ONLY valid JSON - no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000  # Increased token limit
            )

            response_text = response.choices[0].message.content.strip()
            logger.info(f"LLM consolidation response length: {len(response_text)} characters")
            logger.debug(f"LLM consolidation response: {response_text[:500]}...")

            if not response_text:
                logger.error("Empty response from LLM")
                raise ValueError("Empty response from LLM")

            # Strip markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```

            response_text = response_text.strip()

            # Parse and validate the JSON response
            try:
                json_data = json.loads(response_text)
                consolidated = ConsolidatedTaxonomy.model_validate(json_data)
                logger.info(f"Successfully consolidated taxonomy: {len(consolidated.intent_categories)} intents, {len(consolidated.sentiment_categories)} sentiments")

                # Validate indicator uniqueness
                duplicates = self._validate_indicator_uniqueness(consolidated)
                if duplicates:
                    logger.warning(f"⚠️  QUALITY ISSUE: {len(duplicates)} indicators are not unique across categories")
                    logger.warning("This may reduce classification accuracy in production. Consider regenerating or manually editing indicators.")
                else:
                    logger.info("✅ All key indicators are unique across categories")

                return consolidated

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Response text: {response_text}")
                raise
            except Exception as e:
                logger.error(f"Failed to validate consolidated taxonomy: {e}")
                logger.error(f"JSON data: {json_data}")
                raise

        except Exception as e:
            logger.error(f"LLM consolidation request failed: {e}")
            raise

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
            # Use real email examples if available, otherwise generate synthetic
            real_examples = intent_data.get('real_email_examples', [])
            if real_examples:
                for example in real_examples[:3]:
                    # Format real email examples with subject + snippet
                    example_text = f"[{example['subject']}] {example['content'][:150]}..."
                    # Escape quotes in the example text
                    example_text = example_text.replace('"', '\\"')
                    yaml_lines.append(f'      - "{example_text}"')
            else:
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
                f'    business_value: "{sentiment_data.get("business_relevance", "Track customer sentiment patterns")}"',
                f'    coverage: "{coverage_pct:.1f}%"  # {sentiment_data["total_emails"]} emails',
                "    decision_rules:"
            ])

            # Use LLM-derived decision rules, not hardcoded ones
            for rule in sentiment_data.get('decision_rules', []):
                yaml_lines.append(f'      - "{rule}"')

            yaml_lines.append("    key_indicators:")
            # Use LLM-derived indicators
            for indicator in sentiment_data.get('sample_indicators', []):
                yaml_lines.append(f'      - "{indicator}"')

            yaml_lines.append("    examples:")
            # Use real email examples if available, otherwise generate synthetic
            real_examples = sentiment_data.get('real_email_examples', [])
            if real_examples:
                for example in real_examples[:3]:
                    # Format real email examples with subject + snippet
                    example_text = f"[{example['subject']}] {example['content'][:150]}..."
                    # Escape quotes in the example text
                    example_text = example_text.replace('"', '\\"')
                    yaml_lines.append(f'      - "{example_text}"')
            else:
                examples = self._generate_clean_examples(sentiment_name, sentiment_data.get('sample_indicators', []))
                for example in examples:
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

    def generate_system_prompt(self, consolidated_taxonomy: Dict[str, Any]) -> str:
        """Generate production-ready system prompt for NetSuite email classification."""

        intent_categories = consolidated_taxonomy['intent_categories']
        sentiment_categories = consolidated_taxonomy['sentiment_categories']

        # Calculate total emails for coverage percentages
        total_emails = sum(cat.get('total_emails', 0) for cat in intent_categories.values())

        prompt = f"""# NetSuite Collection Email Classification System

You are an expert email classifier for collections operations. Your task is to analyze incoming customer emails and classify them by INTENT and SENTIMENT to help collections teams respond appropriately.

## CLASSIFICATION OVERVIEW

**INTENT**: What the customer wants to achieve (their purpose/goal)
**SENTIMENT**: The customer's emotional tone and communication style

You must classify each email with exactly ONE intent and ONE sentiment category.

## INTENT CATEGORIES

"""

        # Add intent categories with examples and decision rules
        for intent_name, intent_data in intent_categories.items():
            coverage_pct = (intent_data.get('total_emails', 0) / total_emails) * 100 if total_emails > 0 else 0

            prompt += f"""### {intent_name}

**Definition**: {intent_data.get('definition', 'No definition available')}

**When to use**:
"""
            for rule in intent_data.get('decision_rules', [])[:3]:  # Top 3 rules
                prompt += f"- {rule}\n"

            prompt += f"""
**Key indicators**: {', '.join(f'"{ind}"' for ind in intent_data.get('key_indicators', [])[:4])}

**Business impact**: {intent_data.get('business_relevance', 'Standard processing')}

"""

        prompt += f"""
## SENTIMENT CATEGORIES

"""

        # Add sentiment categories
        for sentiment_name, sentiment_data in sentiment_categories.items():
            coverage_pct = (sentiment_data.get('total_emails', 0) / total_emails) * 100 if total_emails > 0 else 0

            prompt += f"""### {sentiment_name}

**Definition**: {sentiment_data.get('definition', 'No definition available')}

**When to use**:
"""
            for rule in sentiment_data.get('decision_rules', [])[:3]:  # Top 3 rules
                prompt += f"- {rule}\n"

            prompt += f"""
**Key indicators**: {', '.join(f'"{ind}"' for ind in sentiment_data.get('key_indicators', [])[:4])}

**Collections impact**: {sentiment_data.get('business_relevance', 'Standard handling')}

"""

        # Add classification instructions and output format
        prompt += f"""
## CLASSIFICATION PROCESS

1. **Read the complete email** including subject line and body content
2. **Identify the primary intent** - what does the customer want?
3. **Determine the sentiment** - how are they communicating?
4. **Apply business context** - consider collections operational needs
5. **Choose the most specific applicable categories**

## OUTPUT FORMAT

Respond with ONLY a valid JSON object in this exact format:

```json
{{
    "intent": "Exact Intent Category Name",
    "sentiment": "Exact Sentiment Category Name",
    "confidence": "high|medium|low",
    "reasoning": "Brief 1-2 sentence explanation of classification decision",
    "key_phrases": ["phrase1", "phrase2", "phrase3"],
    "business_priority": "high|medium|low",
    "suggested_action": "Recommended next step for collections team"
}}
```

## CLASSIFICATION GUIDELINES

**Intent Priority Rules**:
- Focus on the customer's PRIMARY purpose, not secondary mentions
- When multiple intents are present, choose the one requiring immediate action
- Consider the business impact of misclassification

**Sentiment Priority Rules**:
- Prioritize emotionally significant sentiments (frustrated > cooperative > neutral)
- Consider how the sentiment affects required response approach
- Look for subtle emotional indicators beyond obvious language

**Quality Standards**:
- **High confidence**: Clear indicators, unambiguous classification
- **Medium confidence**: Some indicators present, minor ambiguity
- **Low confidence**: Weak indicators, significant uncertainty

**Business Priority Guidelines**:
- **High**: Urgent issues, frustrated customers, payment problems
- **Medium**: Standard requests, cooperative customers, routine updates
- **Low**: Informational only, no immediate action required

## EXAMPLES

"""

        # Add real examples from each major category if available
        example_count = 0
        for intent_name, intent_data in list(intent_categories.items())[:2]:  # Top 2 intents
            real_examples = intent_data.get('real_email_examples', [])

            if real_examples and len(real_examples) > 0:
                # Use first real example
                example = real_examples[0]
                subject = example.get('subject', 'No subject')
                body = example.get('content', 'No content')[:300]  # Limit body length

                prompt += f"""**{intent_name} Example** (Real Email):
```
Subject: {subject}
Body: {body}

Classification:
{{
    "intent": "{intent_name}",
    "sentiment": "Professional",
    "confidence": "high",
    "reasoning": "Email demonstrates clear {intent_name.lower()} intent with professional tone",
    "key_phrases": {intent_data.get('key_indicators', [''])[:3]},
    "business_priority": "medium",
    "suggested_action": "Review and respond according to category guidelines"
}}
```

"""
            else:
                # Fallback to synthetic example
                prompt += f"""**{intent_name} Example**:
```
Subject: Regarding Account Update
Body: Please update your records with our new contact information for future correspondence.

Classification:
{{
    "intent": "{intent_name}",
    "sentiment": "Cooperative",
    "confidence": "high",
    "reasoning": "Clear {intent_name.lower()} request with polite tone",
    "key_phrases": {intent_data.get('key_indicators', [''])[:3]},
    "business_priority": "medium",
    "suggested_action": "Update customer records and acknowledge receipt"
}}
```

"""
            example_count += 1

        prompt += f"""
## CRITICAL REMINDERS

- Respond with ONLY the JSON object - no additional text or formatting
- Use exact category names as defined above
- Ensure JSON is valid and properly formatted
- Consider collections operational context in your decisions
- When uncertain, prioritize business value over perfect categorization

---

*This system prompt was generated from analysis of {total_emails} real customer collection emails*
"""

        return prompt

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
                'decision_rules': sentiment_data.get('decision_rules', []),  # Use LLM data
                'examples': self._generate_clean_examples(sentiment_name, sentiment_data.get('sample_indicators', []))  # Use LLM data
            }

        return structured