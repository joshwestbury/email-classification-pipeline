#!/usr/bin/env python3
"""
LLM analysis module for email taxonomy pipeline.

Analyzes email clusters and proposes categories using OpenAI API.
"""

import json
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from tqdm import tqdm
import re
import random
import hashlib
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .sentiment_analyzer import SentimentAnalyzer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClusterAnalysis(BaseModel):
    """Pydantic model for LLM cluster analysis response validation."""

    proposed_intent: str = Field(..., min_length=1, max_length=200, description="Specific intent category name")
    intent_definition: str = Field(..., min_length=10, max_length=800, description="Precise definition of customer intent")
    proposed_sentiment: str = Field(..., min_length=1, max_length=200, description="Specific emotional tone category name")
    sentiment_definition: str = Field(..., min_length=10, max_length=800, description="Precise definition of emotional state")
    decision_rules: List[str] = Field(..., min_items=3, max_items=6, description="3-6 IF/THEN rules referencing observable cues")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0.0-1.0")
    sample_indicators: List[str] = Field(..., min_items=1, max_items=10, description="Specific phrases indicating this category")
    emotional_markers: List[str] = Field(default_factory=list, max_items=10, description="Emotional indicators in text")
    reasoning: str = Field(..., min_length=20, max_length=1000, description="Detailed explanation of clustering rationale")
    business_relevance: str = Field(..., min_length=10, max_length=500, description="Business value for collections operations")

    @field_validator("decision_rules")
    @classmethod
    def rules_should_have_structure(cls, v: List[str]) -> List[str]:
        """Validate decision rules have clear conditional structure (soft validation)."""
        for i, rule in enumerate(v):
            s = rule.lower()
            # Require explicit IF and THEN (or close synonyms)
            has_if = any(m in s for m in ["if ", "when ", "where "])
            has_then = any(m in s for m in [" then ", " should ", " classify ", " tag ", " leans to "])

            if not (has_if and has_then):
                logger.warning(f"Decision rule {i+1} lacks clear IF/THEN structure: {rule[:100]}")

        return v  # Don't fail validation, just warn

    class Config:
        extra = "forbid"  # Reject any extra fields not defined in schema


class LLMAnalyzer:
    """Analyzes email clusters and proposes categories using LLM."""

    def __init__(self, model: str = "gpt-4o", top_clusters: int = 26, api_key: Optional[str] = None, preanalysis_mode: str = "features"):
        self.model = model
        self.top_clusters = top_clusters
        self.preanalysis_mode = preanalysis_mode

        # Initialize sentiment analyzer with configured mode
        self.sentiment_analyzer = SentimentAnalyzer(mode=preanalysis_mode)

        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _deterministic_seed(cluster_id: str) -> int:
        """Generate deterministic seed from cluster ID for reproducible sampling.

        Uses MD5 hash to ensure consistent results across runs, avoiding Python's
        randomized hash() function which varies per interpreter session.

        Args:
            cluster_id: Cluster identifier string

        Returns:
            Deterministic integer seed for random number generation
        """
        return int.from_bytes(
            hashlib.md5(cluster_id.encode("utf-8")).digest()[:8],
            byteorder="big"
        )

    @staticmethod
    def _normalize_category(label: str) -> str:
        """Normalize category names for consistent tallying.

        Prevents fragmentation like "Payment Plan" vs "payment plan" vs "Payment plan".

        Args:
            label: Raw category label from LLM

        Returns:
            Normalized category name (title case, collapsed whitespace)
        """
        return re.sub(r"\s+", " ", label.strip()).title()

    def get_cluster_samples(self, cluster_analysis: Dict[str, Any], source_data: Dict[str, Any], cluster_id: str, sample_size: int = 5) -> List[Dict[str, Any]]:
        """Get sample emails from a specific cluster for analysis."""
        cluster_info = cluster_analysis.get(cluster_id, {})
        sample_indices = cluster_info.get('sample_indices', [])

        if not sample_indices:
            logger.warning(f"No sample indices found for cluster {cluster_id}")
            return []

        # Get emails from source data
        emails = source_data.get('emails', [])
        sample_emails = []

        for idx in sample_indices[:sample_size]:
            if idx < len(emails):
                email = emails[idx]
                if email.get('direction') == 'incoming':
                    enriched_sample = {
                        'subject': email.get('subject', 'No Subject'),
                        'content': self._clean_email_content(email.get('content', email.get('message', ''))),
                        'thread_id': email.get('thread_id', 'unknown'),
                        'sender': email.get('sender', 'unknown')
                    }
                    sample_emails.append(enriched_sample)

        # If we don't have enough samples from cluster, randomly sample from incoming emails
        if len(sample_emails) < sample_size:
            logger.info(f"Only found {len(sample_emails)} samples for cluster {cluster_id}, filling with random samples")

            incoming_emails = [e for e in emails if e.get('direction') == 'incoming']
            if incoming_emails:
                # Use deterministic seed for reproducible sampling across runs
                random.seed(self._deterministic_seed(cluster_id))
                additional_samples = random.sample(
                    incoming_emails,
                    min(sample_size - len(sample_emails), len(incoming_emails))
                )

                for email in additional_samples:
                    enriched_sample = {
                        'subject': email.get('subject', 'No Subject'),
                        'content': self._clean_email_content(email.get('content', email.get('message', ''))),
                        'thread_id': email.get('thread_id', 'unknown'),
                        'sender': email.get('sender', 'unknown')
                    }
                    sample_emails.append(enriched_sample)

        return sample_emails[:sample_size]

    def _clean_email_content(self, content: str) -> str:
        """Clean and truncate email content for LLM analysis.

        Aggressively removes noise (HTML, signatures, quoted replies, footers)
        that can confuse sentiment and intent classification.
        """
        if not content:
            return "No content available"

        # Strip HTML tags
        content = re.sub(r"<[^>]+>", "", content)

        # Drop quoted reply blocks (common email patterns)
        content = re.sub(r"(?m)^>.*$", "", content)  # Lines starting with >
        content = re.sub(r"(?is)^On .* wrote:.*", "", content)  # "On [date] [person] wrote:"

        # Drop common signature/footer markers (split at first match)
        content = re.split(
            r"(?im)^(thanks[,\s]|regards[,\s]|best[,\s]|sincerely[,\s]|--\s|__+|confidentiality notice|this message.*confidential)",
            content
        )[0]

        # Collapse excessive whitespace
        content = re.sub(r"\s+\n", "\n", content)  # Remove trailing spaces before newlines
        content = re.sub(r"\n{3,}", "\n\n", content).strip()  # Max 2 consecutive newlines

        # Limit length for API efficiency
        max_length = 1200
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=12),
        retry=retry_if_exception_type((
            json.JSONDecodeError,
            KeyError,
            APIError,
            RateLimitError,
            APITimeoutError
        )),
        reraise=True
    )
    def _make_llm_request_with_validation(self, prompt: str, cluster_id: str) -> LLMClusterAnalysis:
        """Make LLM request with validation and retry logic.

        Attempts to use OpenAI's response_format JSON mode for guaranteed valid JSON.
        Falls back to regex extraction if the model doesn't support it.
        """
        try:
            # Try using response_format for JSON mode (gpt-4o, gpt-4-turbo, etc.)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are an expert in customer service and collections email analysis. You help categorize customer communications for business intelligence and automated processing. You MUST respond with valid JSON only - no other text, explanations, or formatting."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1200
                )

                response_text = response.choices[0].message.content.strip()
                logger.debug(f"Raw LLM response for cluster {cluster_id}: {response_text[:200]}...")

                # With response_format, we can parse directly
                raw_json = json.loads(response_text)
                validated_response = LLMClusterAnalysis(**raw_json)

                logger.info(f"Successfully validated LLM response for cluster {cluster_id} (JSON mode)")
                return validated_response

            except (TypeError, Exception) as format_error:
                # Model doesn't support response_format, fall back to regex extraction
                if "response_format" in str(format_error) or "json_object" in str(format_error):
                    logger.debug(f"Model {self.model} doesn't support response_format, using regex fallback")

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert in customer service and collections email analysis. You help categorize customer communications for business intelligence and automated processing. You MUST respond with valid JSON only - no other text, explanations, or formatting."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1200
                    )

                    response_text = response.choices[0].message.content.strip()
                    logger.debug(f"Raw LLM response for cluster {cluster_id}: {response_text[:200]}...")

                    # Extract JSON from response using regex
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if not json_match:
                        raise ValueError(f"No JSON object found in LLM response for cluster {cluster_id}")

                    # Parse and validate JSON
                    raw_json = json.loads(json_match.group())
                    validated_response = LLMClusterAnalysis(**raw_json)

                    logger.info(f"Successfully validated LLM response for cluster {cluster_id} (regex mode)")
                    return validated_response
                else:
                    # Different error, re-raise
                    raise

        except ValidationError as e:
            logger.error(f"Pydantic validation error for cluster {cluster_id}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for cluster {cluster_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for cluster {cluster_id}: {e}")
            raise

    def analyze_cluster_with_llm(self, cluster_id: str, cluster_analysis: Dict[str, Any], source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single cluster using LLM to propose categories."""
        logger.info(f"Analyzing cluster {cluster_id} with LLM...")

        # Get sample emails from cluster
        sample_emails = self.get_cluster_samples(cluster_analysis, source_data, cluster_id, sample_size=5)

        if not sample_emails:
            logger.warning(f"No sample emails found for cluster {cluster_id}")
            return {"error": "No sample emails found for cluster"}

        # Get cluster statistics
        cluster_info = cluster_analysis.get(cluster_id, {})
        cluster_size = cluster_info.get('size', 0)
        cluster_percentage = cluster_info.get('percentage', 0)

        # Get pre-analysis data (features or labels depending on mode)
        linguistic_preanalysis = cluster_info.get('linguistic_preanalysis', {})
        sentiment_analysis = cluster_info.get('sentiment_analysis', {})

        # Prepare prompt with sample emails
        samples_text = ""
        for i, email in enumerate(sample_emails, 1):
            samples_text += f"\nEmail {i}:\n"
            samples_text += f"Subject: {email['subject']}\n"
            samples_text += f"Content: {email['content']}\n"
            samples_text += "-" * 50 + "\n"

        # Prepare pre-analysis section based on mode
        preanalysis_section = ""
        if self.preanalysis_mode == "features" and linguistic_preanalysis:
            # Feature-based pre-analysis (no category labels)
            feature_summary = linguistic_preanalysis.get('feature_summary', {})
            feature_examples = linguistic_preanalysis.get('top_feature_examples', {})

            preanalysis_section = "\n## Linguistic Pre-Analysis (Features)\n\n"
            preanalysis_section += "The following linguistic features have been detected in this cluster:\n\n"
            preanalysis_section += json.dumps({
                'feature_summary': feature_summary,
                'top_feature_examples': feature_examples
            }, indent=2)
            preanalysis_section += "\n\nThese are purely descriptive linguistic measurements. Use them as context, but let the sentiment categories emerge naturally from the email content.\n"

        elif self.preanalysis_mode == "labels" and sentiment_analysis:
            # Label-based pre-analysis (legacy/debug mode)
            dominant_sentiment = sentiment_analysis.get('dominant_sentiment', 'unknown')
            sentiment_dist = sentiment_analysis.get('distribution', {})

            preanalysis_section = "\n## Pattern-Based Sentiment Analysis (Pre-Analysis)\n\n"
            preanalysis_section += f"Dominant Sentiment: {dominant_sentiment}\n"
            preanalysis_section += f"Sentiment Distribution: {sentiment_dist}\n"

        prompt = f"""
        Analyze the following collection of INCOMING CUSTOMER emails that have been clustered together based on semantic similarity. These are emails RECEIVED by a collections/accounts receivable department FROM CUSTOMERS.

        CRITICAL: Create SPECIFIC, GRANULAR categories that capture distinct customer communication patterns.

        ## CONTEXT FOR ANALYSIS

        You are analyzing a cluster of semantically similar customer emails from a collections context. Your goal is to discover the natural categories that emerge from the data itself, not to fit emails into predetermined boxes.

        Cluster Statistics:
        - Cluster ID: {cluster_id}
        - Cluster Size: {cluster_size} incoming customer emails
        - Percentage of Incoming Emails: {cluster_percentage:.1f}%
{preanalysis_section}
        Your task is to:
        1. Identify the natural intent that emerges from these customer emails
        2. Determine the authentic emotional tone present in the communications
        3. Create distinct, meaningful categories useful for collections operations
        4. Suggest clear decision rules based on actual patterns observed
        5. Focus on operational value rather than forced categorization

        Sample CUSTOMER emails from the cluster:
        {samples_text}

        CRITICAL: You must respond with ONLY a valid JSON object. Do not include any text before or after the JSON. Do not use markdown formatting or code blocks. Return only the raw JSON.

        Please provide your analysis in the following exact JSON format:
        {{
            "proposed_intent": "specific intent category name",
            "intent_definition": "precise definition focusing on customer's specific request or communication purpose",
            "proposed_sentiment": "specific emotional tone category name",
            "sentiment_definition": "precise definition of the customer's emotional state and communication style",
            "decision_rules": ["IF ... THEN rule 1", "IF ... THEN rule 2", "IF ... THEN rule 3"],
            "confidence": 0.0,
            "sample_indicators": ["specific phrase 1", "specific phrase 2", "specific phrase 3"],
            "emotional_markers": ["emotional indicator 1", "emotional indicator 2"],
            "reasoning": "detailed explanation of why these emails cluster together",
            "business_relevance": "specific value this category provides to collections operations"
        }}

        IMPORTANT:
        - "confidence" must be a number between 0.0 and 1.0 (e.g., 0.95 for high confidence, 0.7 for medium, 0.4 for low)
        - "decision_rules" should be 3-6 IF/THEN statements referencing observable patterns in the emails

        IMPORTANT FORMATTING RULES:
        - Return ONLY the JSON object
        - No additional text, explanations, or markdown
        - Ensure all strings are properly quoted
        - Ensure all arrays are properly formatted with square brackets
        - Do not use trailing commas
        - Validate that your response is parseable JSON before sending

        ## UNDERSTANDING INTENT AND SENTIMENT (for discovery)

        **INTENT (What outcome the customer wants in the AR/collections process)**
        - **Definition**: The primary, operational outcome the customer is trying to achieve with this email (e.g., change invoice state, obtain payment credit, alter terms/timing, correct records).
        - **Inclusion cues** (choose the single dominant intent):
          • Clear "ask" that would change an AR record, balance, due date, charge, or service status
          • Procedural requests requiring staff action (resend, reissue, attach docs, update entity/contact, unblock/suspend)
          • Information-seeking specifically tied to payment, balance, invoice, credit/adjustment, or service status
        - **Exclusions** (do NOT call these "intent" unless they contain a concrete AR outcome):
          • Pure FYI with no implied change to AR state
          • Relationship talk (thanks/apologies) without an ask affecting AR
          • Forwarded material with no new instruction
        - **Output**: A specific, 1–3 word label that naturally describes the outcome customers are trying to cause (no generic catch-alls).

        **SENTIMENT (How they feel and communicate while pursuing that outcome)**
        - **Definition**: The emotional state and communication style expressed in the message.
        - **Assess on three dimensions** (summarize into one concise sentiment label):
          1) Valence: positive / neutral / negative
          2) Intensity: mild / moderate / strong
          3) Communication style: collaborative, transactional, defensive, impatient, escalatory, accusatory
        - **Linguistic markers** (non-exhaustive examples as evidence):
          • Positive/collaborative: "thanks", "appreciate", "happy to", "we can provide", softeners ("could you", "when you have a moment")
          • Neutral/administrative: declarative statements, formality, no affective words, passive constructions
          • Urgency/escalation: deadlines/dates, "ASAP", repeat punctuation, all-caps/exclamations, "escalate", "legal", "terminate"
          • Frustration/blame: "incorrect", "again", "still waiting", "you charged", "this is unacceptable"
          • Apology/constraint: "sorry", "delay", "we're short", "cannot pay until"
        - **Output**: A specific, 1–3 word sentiment name reflecting the dominant pattern (e.g., "Polite Concern", "Neutral Professional", "Escalated Frustration"). When mixed, pick the sentiment that most changes handling for collections.

        **DECISION RULES (make these operational and testable)**
        - Provide 3–6 IF/THEN rules referencing observable cues. Example:
          • IF the email contains a payment verb ("pay", "remit", "wire", "schedule") AND a future date expression, THEN intent leans to "Payment Scheduling".
          • IF contains dispute terms ("incorrect", "dispute", "overcharged", "credit due") OR requests for corrected docs, THEN intent leans to "Invoice Correction/Dispute".
          • IF contains deadlines, escalatory language, or termination threats, THEN sentiment leans to "Escalated Frustration".

        **SAMPLE INDICATORS**
        - Provide 3–8 **verbatim** short phrases from the samples that support the chosen intent.
        - Provide 2–6 **verbatim** phrases that support the chosen sentiment (emotional markers).

        **BUSINESS RELEVANCE**
        - Describe how this intent+sentiment combo changes queueing, SLA, or playbook (e.g., route to Disputes, prioritize within 24h, require attachment checklist, trigger manager review).
        """

        try:
            # Use retry-enabled LLM request with validation
            validated_response = self._make_llm_request_with_validation(prompt, cluster_id)

            # Convert validated pydantic model to dict and add metadata
            analysis_result = validated_response.model_dump()
            analysis_result['cluster_id'] = cluster_id
            analysis_result['cluster_size'] = cluster_size
            analysis_result['cluster_percentage'] = cluster_percentage
            analysis_result['sample_count'] = len(sample_emails)
            analysis_result['validation_status'] = 'success'

            # Add provenance fields for auditing
            analysis_result['provenance'] = {
                'model': self.model,
                'preanalysis_mode': self.preanalysis_mode,
                'prompt_bytes': len(prompt.encode('utf-8')),
                'sample_emails_analyzed': len(sample_emails)
            }

            logger.info(f"Successfully analyzed cluster {cluster_id} with validation")
            logger.debug(f"First decision rule: {validated_response.decision_rules[0][:100]}")
            logger.debug(f"Top indicator: {validated_response.sample_indicators[0] if validated_response.sample_indicators else 'none'}")

            return analysis_result

        except ValidationError as e:
            logger.error(f"Final validation failure for cluster {cluster_id} after retries: {e}")
            return {
                "error": f"Validation failed after retries: {str(e)}",
                "cluster_id": cluster_id,
                "validation_status": "failed",
                "error_type": "validation_error"
            }
        except json.JSONDecodeError as e:
            logger.error(f"Final JSON decode failure for cluster {cluster_id} after retries: {e}")
            return {
                "error": f"JSON decode failed after retries: {str(e)}",
                "cluster_id": cluster_id,
                "validation_status": "failed",
                "error_type": "json_decode_error"
            }
        except Exception as e:
            logger.error(f"Unexpected error analyzing cluster {cluster_id}: {str(e)}")
            return {
                "error": f"Unexpected error: {str(e)}",
                "cluster_id": cluster_id,
                "validation_status": "failed",
                "error_type": "unexpected_error"
            }

    def analyze_clusters(self, cluster_results: Dict[str, Any], source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze top clusters and propose taxonomy categories."""
        logger.info("Starting LLM cluster analysis...")

        cluster_analysis = cluster_results.get('cluster_analysis', {})
        selected_clusters = cluster_results.get('top_clusters', [])

        # Limit to our maximum if more clusters were selected
        if len(selected_clusters) > self.top_clusters:
            logger.info(f"Limiting analysis from {len(selected_clusters)} to {self.top_clusters} clusters")
            selected_clusters = selected_clusters[:self.top_clusters]

        if not selected_clusters:
            logger.warning("No clusters found for analysis")
            return {"error": "No clusters available for analysis"}

        logger.info(f"Analyzing {len(selected_clusters)} selected clusters (multi-criteria selection): {selected_clusters}")

        # Enrich clusters with sentiment analysis
        logger.info("Enriching clusters with pattern-based sentiment analysis...")
        emails = source_data.get('emails', [])
        enriched_cluster_analysis = self.sentiment_analyzer.enrich_clusters_with_sentiment(cluster_analysis, emails)

        # Save enriched cluster analysis for reference
        logger.info("Sentiment enrichment completed - enriched data will be used for LLM analysis")

        # Analyze each cluster
        cluster_analyses = {}
        total_emails_analyzed = 0

        for cluster_id in tqdm(selected_clusters, desc="Analyzing clusters"):
            cluster_info = cluster_analysis.get(cluster_id, {})
            cluster_size = cluster_info.get('size', 0)

            analysis = self.analyze_cluster_with_llm(cluster_id, enriched_cluster_analysis, source_data)
            cluster_analyses[cluster_id] = analysis

            if 'error' not in analysis:
                total_emails_analyzed += cluster_size

        # Generate summary statistics with validation tracking
        total_emails = sum(info.get('size', 0) for info in cluster_analysis.values() if info.get('cluster_id', -1) != -1)
        coverage_percentage = (total_emails_analyzed / total_emails * 100) if total_emails > 0 else 0

        # Count validation results
        successful_validations = len([a for a in cluster_analyses.values() if a.get('validation_status') == 'success'])
        failed_validations = len([a for a in cluster_analyses.values() if a.get('validation_status') == 'failed'])
        validation_success_rate = (successful_validations / len(cluster_analyses) * 100) if cluster_analyses else 0

        # Compile proposed categories with normalized names
        intent_categories = {}
        sentiment_categories = {}

        for cluster_id, analysis in cluster_analyses.items():
            if 'error' not in analysis:
                # Normalize category names to prevent fragmentation
                intent = self._normalize_category(analysis.get('proposed_intent', ''))
                sentiment = self._normalize_category(analysis.get('proposed_sentiment', ''))

                if intent:
                    if intent not in intent_categories:
                        intent_categories[intent] = {
                            'definition': analysis.get('intent_definition', ''),
                            'clusters': [],
                            'total_emails': 0
                        }
                    intent_categories[intent]['clusters'].append(cluster_id)
                    intent_categories[intent]['total_emails'] += analysis.get('cluster_size', 0)

                if sentiment:
                    if sentiment not in sentiment_categories:
                        sentiment_categories[sentiment] = {
                            'definition': analysis.get('sentiment_definition', ''),
                            'clusters': [],
                            'total_emails': 0
                        }
                    sentiment_categories[sentiment]['clusters'].append(cluster_id)
                    sentiment_categories[sentiment]['total_emails'] += analysis.get('cluster_size', 0)

        results = {
            'analysis_summary': {
                'clusters_analyzed': len(selected_clusters),
                'clusters_selected_by_criteria': 'all_clusters (comprehensive analysis)',
                'successful_analyses': len([a for a in cluster_analyses.values() if 'error' not in a]),
                'successful_validations': successful_validations,
                'failed_validations': failed_validations,
                'validation_success_rate': round(validation_success_rate, 1),
                'total_emails_in_analyzed_clusters': total_emails_analyzed,
                'coverage_percentage': round(coverage_percentage, 1),
                'model_used': self.model
            },
            'cluster_analyses': cluster_analyses,
            'proposed_taxonomy': {
                'intent_categories': intent_categories,
                'sentiment_categories': sentiment_categories
            },
            'clustering_stats': cluster_results.get('cluster_stats', {}),
            'enriched_cluster_analysis': enriched_cluster_analysis,
            'configuration': {
                'model': self.model,
                'top_clusters_analyzed': self.top_clusters
            }
        }

        logger.info(f"LLM analysis complete: {len(intent_categories)} intent categories, {len(sentiment_categories)} sentiment categories")
        logger.info(f"Coverage: {coverage_percentage:.1f}% of emails ({total_emails_analyzed}/{total_emails})")
        logger.info(f"Validation: {validation_success_rate:.1f}% success rate ({successful_validations}/{len(cluster_analyses)} clusters)")

        return results

    def save_results(self, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save analysis results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved LLM analysis results to {output_path}")