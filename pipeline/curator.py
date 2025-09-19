#!/usr/bin/env python3
"""
Taxonomy curation module for email taxonomy pipeline.

Converts LLM analysis results into final curated taxonomy files.
"""

import json
import yaml
from typing import Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TaxonomyCurator:
    """Curates LLM analysis results into final taxonomy files."""

    def __init__(self):
        self.curation_rules = {
            # Intent consolidation rules
            'intent_consolidation': {
                'Payment Inquiry': ['payment inquiry', 'payment status', 'payment question'],
                'Invoice Management': ['invoice management', 'invoice question', 'billing inquiry'],
                'Information Request': ['information request', 'general inquiry', 'account inquiry']
            },
            # Sentiment consolidation rules
            'sentiment_consolidation': {
                'Cooperative': ['cooperative', 'willing', 'collaborative'],
                'Administrative': ['administrative', 'neutral', 'business'],
                'Informational': ['informational', 'factual', 'inquiry'],
                'Frustrated': ['frustrated', 'angry', 'dissatisfied']
            }
        }

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

        guide = f"""# Email Taxonomy Labeling Guide

Generated by Email Taxonomy Discovery Pipeline

## Overview

This guide provides detailed instructions for classifying customer emails into intent and sentiment categories based on analysis of {taxonomy['metadata']['total_emails_analyzed']} emails with {taxonomy['metadata']['coverage_percentage']:.1f}% coverage.

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

**Definition**: {intent_data['definition']}

**Coverage**: {intent_data['coverage']['total_emails']} emails across {intent_data['coverage']['clusters']} clusters

**Decision Rules**:
"""
            for rule in intent_data['decision_rules']:
                guide += f"- {rule}\n"

            guide += "\n**Examples**:\n"
            for example in intent_data['examples']:
                guide += f"- \"{example}\"\n"

            guide += "\n"

        guide += "\n## Sentiment Categories\n\n"

        # Add sentiment categories
        for sentiment_name, sentiment_data in taxonomy['sentiment_categories'].items():
            guide += f"""### {sentiment_name}

**Definition**: {sentiment_data['definition']}

**Coverage**: {sentiment_data['coverage']['total_emails']} emails across {sentiment_data['coverage']['clusters']} clusters

**Decision Rules**:
"""
            for rule in sentiment_data['decision_rules']:
                guide += f"- {rule}\n"

            guide += "\n**Examples**:\n"
            for example in sentiment_data['examples']:
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

        # Consolidate categories
        consolidated_taxonomy = self.consolidate_categories(llm_analysis)

        # Generate final taxonomy
        analysis_summary = llm_analysis.get('analysis_summary', {})
        final_taxonomy = self.generate_taxonomy_yaml(consolidated_taxonomy, analysis_summary)

        # Generate labeling guide
        labeling_guide = self.generate_labeling_guide(final_taxonomy)

        results = {
            'final_taxonomy': final_taxonomy,
            'labeling_guide': labeling_guide,
            'curation_stats': {
                'original_intent_categories': len(llm_analysis.get('proposed_taxonomy', {}).get('intent_categories', {})),
                'final_intent_categories': len(final_taxonomy['intent_categories']),
                'original_sentiment_categories': len(llm_analysis.get('proposed_taxonomy', {}).get('sentiment_categories', {})),
                'final_sentiment_categories': len(final_taxonomy['sentiment_categories']),
                'total_emails_covered': analysis_summary.get('total_emails_in_analyzed_clusters', 0),
                'coverage_percentage': analysis_summary.get('coverage_percentage', 0)
            }
        }

        logger.info(f"Curation complete: {len(final_taxonomy['intent_categories'])} intent + {len(final_taxonomy['sentiment_categories'])} sentiment categories")

        return results

    def save_results(self, curation_results: Dict[str, Any], output_dir: Path) -> None:
        """Save curated taxonomy files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save taxonomy.yaml
        taxonomy_path = output_dir / 'taxonomy.yaml'
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
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