#!/usr/bin/env python3
"""
System prompt generator for LLM email classification.
Transforms taxonomy.yaml into a highly calibrated prompt for production use.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class PromptGenerator:
    """Generate optimized system prompts from taxonomy definitions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt generator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Prompt generation parameters
        self.include_examples = self.config.get('include_examples', True)
        self.include_confidence_scoring = self.config.get('include_confidence_scoring', True)
        self.include_entity_extraction = self.config.get('include_entity_extraction', True)
        self.include_chain_of_thought = self.config.get('include_chain_of_thought', True)
        self.max_examples_per_category = self.config.get('max_examples_per_category', 3)

    def generate(self, taxonomy_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a system prompt from taxonomy YAML.

        Args:
            taxonomy_path: Path to taxonomy.yaml file
            output_path: Optional path to save the generated prompt

        Returns:
            Dictionary containing the system prompt and metadata
        """
        self.logger.info(f"Generating system prompt from {taxonomy_path}")

        # Load taxonomy
        with open(taxonomy_path, 'r') as f:
            taxonomy = yaml.safe_load(f)

        # Generate prompt components
        prompt_sections = []

        # 1. Core instruction
        prompt_sections.append(self._generate_core_instruction())

        # 2. Task definition
        prompt_sections.append(self._generate_task_definition(taxonomy))

        # 3. Intent categories with enhanced definitions
        prompt_sections.append(self._generate_intent_section(taxonomy))

        # 4. Sentiment categories with enhanced definitions
        prompt_sections.append(self._generate_sentiment_section(taxonomy))

        # 5. Classification methodology
        prompt_sections.append(self._generate_methodology_section(taxonomy))

        # 6. Output format specification
        prompt_sections.append(self._generate_output_format(taxonomy))

        # 7. Examples section (if enabled)
        if self.include_examples:
            prompt_sections.append(self._generate_examples_section(taxonomy))

        # 8. Edge cases and disambiguation rules
        prompt_sections.append(self._generate_disambiguation_rules(taxonomy))

        # 9. Final instructions
        prompt_sections.append(self._generate_final_instructions())

        # Combine all sections
        system_prompt = "\n\n".join(prompt_sections)

        # Create structured output
        result = {
            'system_prompt': system_prompt,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'taxonomy_version': taxonomy.get('version', '1.0'),
                'source_emails': taxonomy.get('source_emails', 0),
                'configuration': {
                    'include_examples': self.include_examples,
                    'include_confidence_scoring': self.include_confidence_scoring,
                    'include_entity_extraction': self.include_entity_extraction,
                    'include_chain_of_thought': self.include_chain_of_thought
                }
            },
            'json_schema': self._generate_json_schema(taxonomy)
        }

        # Save if output path provided
        if output_path:
            # Save system prompt as text
            prompt_file = output_path.with_suffix('.txt')
            with open(prompt_file, 'w') as f:
                f.write(system_prompt)
            self.logger.info(f"Saved system prompt to {prompt_file}")

            # Save full result with metadata as JSON
            json_file = output_path.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"Saved prompt metadata to {json_file}")

        return result

    def _generate_core_instruction(self) -> str:
        """Generate the core instruction section."""
        return """# EMAIL CLASSIFICATION SYSTEM PROMPT

You are an expert email classification system specializing in customer collection communications. Your task is to analyze incoming customer emails and classify them with high accuracy according to defined categories.

## CRITICAL INSTRUCTIONS:
1. You MUST classify every email with exactly ONE intent and exactly ONE sentiment
2. Base your classification ONLY on the email content, not assumptions
3. Use the provided decision rules and key indicators systematically
4. When uncertain, apply the disambiguation rules provided
5. Provide confidence scores to indicate classification certainty"""

    def _generate_task_definition(self, taxonomy: Dict[str, Any]) -> str:
        """Generate the task definition section."""
        return f"""## TASK DEFINITION

Analyze the provided customer email and determine:
1. **INTENT** - The primary purpose or goal of the customer's communication
2. **SENTIMENT** - The emotional tone and cooperative level of the message
3. **MODIFIERS** - Any special flags requiring attention (urgency, escalation, payment commitment)
4. **ENTITIES** - Extract relevant business entities (invoice numbers, amounts, dates)
5. **SUMMARY** - Concise summary of the email content (max 2 sentences)
6. **RECOMMENDED ACTION** - Suggested next step for the collection agent

This classification system was derived from analysis of {taxonomy.get('source_emails', 'N/A')} real collection emails."""

    def _generate_intent_section(self, taxonomy: Dict[str, Any]) -> str:
        """Generate enhanced intent categories section."""
        section = "## INTENT CATEGORIES\n\n"
        section += "Classify the email into exactly ONE of these intent categories:\n\n"

        intent_cats = taxonomy.get('intent_categories', {})

        for cat_id, cat_data in intent_cats.items():
            section += f"### {cat_data['display_name'].upper()}\n"
            section += f"**Internal ID:** `{cat_id}`\n"
            section += f"**Definition:** {cat_data['description']}\n\n"

            # Decision rules
            if 'decision_rules' in cat_data:
                section += "**Decision Rules:**\n"
                for rule in cat_data['decision_rules']:
                    section += f"- {rule}\n"
                section += "\n"

            # Key indicators
            if 'key_indicators' in cat_data:
                section += "**Key Phrases to Look For:**\n"
                section += f"```\n{', '.join([f'"{ind}"' for ind in cat_data['key_indicators']])}\n```\n\n"

            # Examples (limited)
            if self.include_examples and 'examples' in cat_data:
                section += "**Example Emails:**\n"
                for i, example in enumerate(cat_data['examples'][:self.max_examples_per_category], 1):
                    section += f"{i}. \"{example}\"\n"
                section += "\n"

            # Business context
            if 'business_value' in cat_data:
                section += f"**Business Context:** {cat_data['business_value']}\n\n"

            section += "---\n\n"

        return section

    def _generate_sentiment_section(self, taxonomy: Dict[str, Any]) -> str:
        """Generate enhanced sentiment categories section."""
        section = "## SENTIMENT CATEGORIES\n\n"
        section += "Classify the emotional tone into exactly ONE of these sentiment categories:\n\n"

        sentiment_cats = taxonomy.get('sentiment_categories', {})

        for cat_id, cat_data in sentiment_cats.items():
            section += f"### {cat_data['display_name'].upper()}\n"
            section += f"**Internal ID:** `{cat_id}`\n"
            section += f"**Definition:** {cat_data['description']}\n\n"

            # Decision rules
            if 'decision_rules' in cat_data:
                section += "**Decision Rules:**\n"
                for rule in cat_data['decision_rules']:
                    section += f"- {rule}\n"
                section += "\n"

            # Key indicators
            if 'key_indicators' in cat_data:
                section += "**Key Emotional Indicators:**\n"
                section += f"```\n{', '.join([f'"{ind}"' for ind in cat_data['key_indicators']])}\n```\n\n"

            # Examples
            if self.include_examples and 'examples' in cat_data:
                section += "**Example Expressions:**\n"
                for i, example in enumerate(cat_data['examples'][:self.max_examples_per_category], 1):
                    section += f"{i}. \"{example}\"\n"
                section += "\n"

            # Special handling notes
            if cat_data.get('coverage', '') == "0%":
                section += "**Note:** This category may be rare but is included for completeness. Apply only when clear indicators are present.\n\n"

            section += "---\n\n"

        return section

    def _generate_methodology_section(self, taxonomy: Dict[str, Any]) -> str:
        """Generate classification methodology section."""
        section = "## CLASSIFICATION METHODOLOGY\n\n"

        if self.include_chain_of_thought:
            section += """### Step-by-Step Classification Process:

1. **Initial Read:** Read the entire email to understand context
2. **Intent Analysis:**
   - Identify the primary ask or purpose
   - Look for key indicator phrases
   - Apply decision rules in order
3. **Sentiment Analysis:**
   - Assess overall emotional tone
   - Look for cooperation indicators
   - Check for frustration or urgency markers
4. **Modifier Detection:**
   - Scan for urgency indicators
   - Check for escalation language
   - Identify payment commitments
5. **Entity Extraction:**
   - Extract invoice numbers (format: INV-XXXX, #XXXXX)
   - Extract payment amounts ($X,XXX.XX)
   - Extract dates (various formats)
6. **Confidence Assessment:**
   - High (>0.8): Clear indicators, no ambiguity
   - Medium (0.5-0.8): Some indicators present
   - Low (<0.5): Weak indicators, needs review

"""

        # Add modifier flags
        modifiers = taxonomy.get('modifier_flags', {})
        if modifiers:
            section += "### MODIFIER FLAGS\n\n"
            section += "Check for these optional modifiers (can have multiple):\n\n"

            for mod_id, mod_data in modifiers.items():
                section += f"**{mod_id.upper()}**\n"
                section += f"- {mod_data['description']}\n"
                section += f"- Indicators: {', '.join([f'"{ind}"' for ind in mod_data['indicators']])}\n\n"

        return section

    def _generate_output_format(self, taxonomy: Dict[str, Any]) -> str:
        """Generate output format specification."""
        section = "## OUTPUT FORMAT\n\n"
        section += "Return your classification in the following JSON structure:\n\n"
        section += """```json
{
  "intent": {
    "category": "payment_inquiry|invoice_management|information_request",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification decision"
  },
  "sentiment": {
    "category": "cooperative|administrative|informational|frustrated",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of sentiment assessment"
  },
  "modifiers": {
    "urgency": boolean,
    "escalation": boolean,
    "payment_commitment": boolean
  },
  "extracted_entities": {
    "invoice_numbers": ["INV-123", "#456"],
    "amounts": ["$1,234.56"],
    "dates": ["2024-01-15", "next week"],
    "company_names": ["Acme Corp"],
    "contact_names": ["John Doe"]
  },
  "summary": "1-2 sentence summary of the email content",
  "recommended_action": "Specific next step for the collection agent",
  "requires_human_review": boolean,
  "review_reason": "Only if requires_human_review is true"
}
```"""

        if self.include_confidence_scoring:
            section += """

### Confidence Scoring Guidelines:
- **0.9-1.0:** Multiple clear indicators present, unambiguous classification
- **0.7-0.89:** Primary indicators present with minor ambiguity
- **0.5-0.69:** Some indicators present but context is unclear
- **0.0-0.49:** Weak or conflicting indicators - flag for human review"""

        return section

    def _generate_examples_section(self, taxonomy: Dict[str, Any]) -> str:
        """Generate annotated examples section."""
        section = "## ANNOTATED EXAMPLES\n\n"
        section += "Here are complete classification examples with reasoning:\n\n"

        # Create synthetic examples combining different intents and sentiments
        examples = [
            {
                "email": "Can you provide an update on case #12345? Payment was sent last week.",
                "intent": "payment_inquiry",
                "sentiment": "cooperative",
                "reasoning": "Asks for payment status (intent) and provides helpful information (sentiment)"
            },
            {
                "email": "Please cancel invoice INV-789. This was already paid under invoice INV-456.",
                "intent": "invoice_management",
                "sentiment": "administrative",
                "reasoning": "Requests invoice cancellation (intent) with factual explanation (sentiment)"
            },
            {
                "email": "This is the third time I'm asking - when will my payment be processed?!",
                "intent": "payment_inquiry",
                "sentiment": "frustrated",
                "reasoning": "Asks about payment timing (intent) with clear frustration markers (sentiment)"
            }
        ]

        for i, ex in enumerate(examples, 1):
            section += f"### Example {i}:\n"
            section += f"**Email:** \"{ex['email']}\"\n\n"
            section += f"**Classification:**\n"
            section += f"- Intent: `{ex['intent']}`\n"
            section += f"- Sentiment: `{ex['sentiment']}`\n"
            section += f"- Reasoning: {ex['reasoning']}\n\n"

        return section

    def _generate_disambiguation_rules(self, taxonomy: Dict[str, Any]) -> str:
        """Generate disambiguation rules for edge cases."""
        return """## DISAMBIGUATION RULES

### When Multiple Intents Appear:
1. **Payment + Invoice:** If both payment and invoice topics appear, prioritize based on the primary question/request
2. **Information + Specific Category:** If asking for information about payments/invoices, classify as payment_inquiry or invoice_management respectively
3. **Default to Most Specific:** Always choose the most specific applicable category

### When Sentiment is Mixed:
1. **Frustration + Cooperation:** If customer shows frustration but still provides information, classify as 'frustrated'
2. **Administrative + Cooperative:** If purely factual but includes helpful elements, classify as 'cooperative'
3. **Neutral Tone:** Default to 'administrative' for purely factual communications

### Special Cases:
- **Out of Office/Automated Replies:** Classify as information_request + informational
- **Thank You Only Emails:** Classify as information_request + cooperative
- **Forward without Comment:** Analyze the forwarded content, not the act of forwarding
- **Multiple Questions:** Focus on the primary or most urgent question

### Confidence Adjustment Rules:
- Reduce confidence by 0.2 if email is very short (<20 words)
- Reduce confidence by 0.3 if multiple valid categories apply
- Set max confidence to 0.7 if no key indicators are present
- Always flag for review if confidence < 0.5 for either classification"""

    def _generate_final_instructions(self) -> str:
        """Generate final instructions section."""
        return """## FINAL INSTRUCTIONS

1. **Accuracy over Speed:** Take time to analyze thoroughly
2. **Use Evidence:** Base decisions on specific phrases and context
3. **Document Reasoning:** Always explain your classification logic
4. **Flag Uncertainty:** Set requires_human_review=true when confidence is low
5. **Be Consistent:** Apply the same rules uniformly across all emails

Remember: This classification directly impacts customer service quality and collection efficiency. Your accurate classification helps agents provide better, faster responses to customers."""

    def _generate_json_schema(self, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON schema for structured output validation."""

        # Get category enums
        intent_categories = list(taxonomy.get('intent_categories', {}).keys())
        sentiment_categories = list(taxonomy.get('sentiment_categories', {}).keys())

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["intent", "sentiment", "modifiers", "extracted_entities", "summary", "recommended_action"],
            "properties": {
                "intent": {
                    "type": "object",
                    "required": ["category", "confidence", "reasoning"],
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": intent_categories
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "reasoning": {
                            "type": "string",
                            "maxLength": 200
                        }
                    }
                },
                "sentiment": {
                    "type": "object",
                    "required": ["category", "confidence", "reasoning"],
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": sentiment_categories
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "reasoning": {
                            "type": "string",
                            "maxLength": 200
                        }
                    }
                },
                "modifiers": {
                    "type": "object",
                    "required": ["urgency", "escalation", "payment_commitment"],
                    "properties": {
                        "urgency": {"type": "boolean"},
                        "escalation": {"type": "boolean"},
                        "payment_commitment": {"type": "boolean"}
                    }
                },
                "extracted_entities": {
                    "type": "object",
                    "properties": {
                        "invoice_numbers": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "amounts": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "dates": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "company_names": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "contact_names": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "summary": {
                    "type": "string",
                    "maxLength": 500
                },
                "recommended_action": {
                    "type": "string",
                    "maxLength": 200
                },
                "requires_human_review": {
                    "type": "boolean"
                },
                "review_reason": {
                    "type": "string",
                    "maxLength": 200
                }
            }
        }

        return schema