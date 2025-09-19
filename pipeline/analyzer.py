#!/usr/bin/env python3
"""
LLM analysis module for email taxonomy pipeline.

Analyzes email clusters and proposes categories using OpenAI API.
"""

import json
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI
from tqdm import tqdm
import re
import random
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Analyzes email clusters and proposes categories using LLM."""

    def __init__(self, model: str = "gpt-4o", top_clusters: int = 8, api_key: Optional[str] = None):
        self.model = model
        self.top_clusters = top_clusters

        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key)

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
                # Use cluster_id as seed for consistency
                random.seed(int(cluster_id) if cluster_id.isdigit() else hash(cluster_id))
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
        """Clean and truncate email content for LLM analysis."""
        if not content:
            return "No content available"

        # Remove HTML tags if any remain
        content = re.sub(r'<[^>]+>', '', content)

        # Limit length for API efficiency
        max_length = 500
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return content.strip()

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

        # Prepare prompt with sample emails
        samples_text = ""
        for i, email in enumerate(sample_emails, 1):
            samples_text += f"\nEmail {i}:\n"
            samples_text += f"Subject: {email['subject']}\n"
            samples_text += f"Content: {email['content']}\n"
            samples_text += "-" * 50 + "\n"

        prompt = f"""
        Analyze the following collection of customer emails that have been clustered together based on semantic similarity. These are emails received by a collections/accounts receivable department.

        Cluster Statistics:
        - Cluster ID: {cluster_id}
        - Cluster Size: {cluster_size} emails
        - Percentage of Dataset: {cluster_percentage:.1f}%

        Your task is to:
        1. Identify the common intent/purpose of these emails
        2. Determine the overall sentiment/tone
        3. Propose a category name and definition
        4. Suggest decision rules for classifying similar emails

        Sample emails from the cluster:
        {samples_text}

        Please provide your analysis in the following JSON format:
        {{
            "proposed_intent": "intent category name",
            "intent_definition": "clear definition of what this intent represents",
            "proposed_sentiment": "sentiment category name",
            "sentiment_definition": "clear definition of what this sentiment represents",
            "decision_rules": ["rule 1", "rule 2", "rule 3"],
            "confidence": "high/medium/low",
            "sample_indicators": ["key phrase 1", "key phrase 2"],
            "reasoning": "explanation of why these emails cluster together",
            "business_relevance": "how this category helps collections teams"
        }}

        Focus on collection-specific intents like:
        - Payment inquiry/status requests
        - Invoice/billing questions
        - Dispute/disagreement
        - Payment promise/commitment
        - Hardship/financial difficulty
        - Information requests
        - Acknowledgment/confirmation

        For sentiment, consider:
        - Cooperative/willing to resolve
        - Frustrated/angry
        - Apologetic/regretful
        - Administrative/neutral
        - Evasive/defensive
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in customer service and collections email analysis. You help categorize customer communications for business intelligence and automated processing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            # Parse JSON response
            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())

                # Add cluster metadata
                analysis_result['cluster_id'] = cluster_id
                analysis_result['cluster_size'] = cluster_size
                analysis_result['cluster_percentage'] = cluster_percentage
                analysis_result['sample_count'] = len(sample_emails)

                return analysis_result
            else:
                logger.error(f"Could not extract JSON from LLM response for cluster {cluster_id}")
                return {"error": "Failed to parse LLM response", "raw_response": response_text}

        except Exception as e:
            logger.error(f"Error analyzing cluster {cluster_id}: {str(e)}")
            return {"error": str(e)}

    def analyze_clusters(self, cluster_results: Dict[str, Any], source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze top clusters and propose taxonomy categories."""
        logger.info("Starting LLM cluster analysis...")

        cluster_analysis = cluster_results.get('cluster_analysis', {})
        top_clusters = cluster_results.get('top_clusters', [])[:self.top_clusters]

        if not top_clusters:
            logger.warning("No clusters found for analysis")
            return {"error": "No clusters available for analysis"}

        logger.info(f"Analyzing top {len(top_clusters)} clusters: {top_clusters}")

        # Analyze each cluster
        cluster_analyses = {}
        total_emails_analyzed = 0

        for cluster_id in tqdm(top_clusters, desc="Analyzing clusters"):
            cluster_info = cluster_analysis.get(cluster_id, {})
            cluster_size = cluster_info.get('size', 0)

            analysis = self.analyze_cluster_with_llm(cluster_id, cluster_analysis, source_data)
            cluster_analyses[cluster_id] = analysis

            if 'error' not in analysis:
                total_emails_analyzed += cluster_size

        # Generate summary statistics
        total_emails = sum(info.get('size', 0) for info in cluster_analysis.values() if info.get('cluster_id', -1) != -1)
        coverage_percentage = (total_emails_analyzed / total_emails * 100) if total_emails > 0 else 0

        # Compile proposed categories
        intent_categories = {}
        sentiment_categories = {}

        for cluster_id, analysis in cluster_analyses.items():
            if 'error' not in analysis:
                intent = analysis.get('proposed_intent', '')
                sentiment = analysis.get('proposed_sentiment', '')

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
                'clusters_analyzed': len(top_clusters),
                'successful_analyses': len([a for a in cluster_analyses.values() if 'error' not in a]),
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
            'configuration': {
                'model': self.model,
                'top_clusters_analyzed': self.top_clusters
            }
        }

        logger.info(f"LLM analysis complete: {len(intent_categories)} intent categories, {len(sentiment_categories)} sentiment categories")
        logger.info(f"Coverage: {coverage_percentage:.1f}% of emails ({total_emails_analyzed}/{total_emails})")

        return results

    def save_results(self, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save analysis results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved LLM analysis results to {output_path}")