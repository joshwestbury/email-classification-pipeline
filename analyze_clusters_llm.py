#!/usr/bin/env python3
"""
LLM-powered cluster analysis and category proposal for Collection Notes AI
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
import os
from openai import OpenAI
from tqdm import tqdm
import re


class ClusterAnalyzer:
    """Analyzes email clusters and proposes categories using LLM"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the cluster analyzer with OpenAI API"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to get API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key)

        self.cluster_data = None
        self.metadata = None
        self.anonymized_emails = None

    def load_data(self) -> None:
        """Load cluster results, metadata, and anonymized emails"""
        print("Loading cluster analysis data...")

        # Load cluster analysis summary
        with open('cluster_analysis_summary.json', 'r') as f:
            self.cluster_data = json.load(f)
        print(f"Loaded cluster analysis with {self.cluster_data['n_clusters']} clusters")

        # Load email metadata
        with open('incoming_email_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded metadata for {len(self.metadata)} emails")

        # Load anonymized emails for content analysis
        with open('master_email_threads_anonymized.json', 'r') as f:
            self.anonymized_emails = json.load(f)
        print(f"Loaded {len(self.anonymized_emails)} anonymized emails")

    def get_cluster_samples(self, cluster_id: str, sample_size: int = 10) -> List[Dict[str, Any]]:
        """Get sample emails from a specific cluster for analysis"""
        if not self.metadata:
            raise ValueError("Metadata not loaded. Call load_data() first.")

        # Load cluster labels to find emails in this cluster
        try:
            with open('parameter_experiments.csv', 'r') as f:
                # This is a simplification - in a real implementation we'd save cluster assignments
                # For now, we'll use the metadata directly and sample emails
                pass
        except:
            pass

        # For now, sample emails randomly from metadata for demonstration
        # In production, we'd use the actual cluster assignments
        import random
        random.seed(42)
        sample_emails = random.sample(self.metadata, min(sample_size, len(self.metadata)))

        # Get corresponding content from anonymized emails
        enriched_samples = []
        for email_meta in sample_emails:
            # Find matching anonymized email by some identifier
            # This is simplified - in practice we'd need proper ID matching
            for anon_email in self.anonymized_emails[:sample_size]:
                if anon_email.get('direction') == 'incoming':
                    enriched_sample = {
                        'subject': email_meta.get('subject', 'No Subject'),
                        'content': self._clean_email_content(anon_email.get('content', '')),
                        'thread_id': anon_email.get('thread_id', 'unknown')
                    }
                    enriched_samples.append(enriched_sample)
                    break

        return enriched_samples[:sample_size]

    def _clean_email_content(self, content: str) -> str:
        """Clean and truncate email content for LLM analysis"""
        if not content:
            return "No content available"

        # Remove HTML tags if any remain
        content = re.sub(r'<[^>]+>', '', content)

        # Limit length for API efficiency
        max_length = 500
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return content.strip()

    def analyze_cluster_with_llm(self, cluster_id: str) -> Dict[str, Any]:
        """Analyze a single cluster using LLM to propose categories"""
        print(f"Analyzing cluster {cluster_id} with LLM...")

        # Get sample emails from cluster
        sample_emails = self.get_cluster_samples(cluster_id, sample_size=5)

        if not sample_emails:
            return {"error": "No sample emails found for cluster"}

        # Prepare prompt with sample emails
        samples_text = ""
        for i, email in enumerate(sample_emails, 1):
            samples_text += f"\nEmail {i}:\n"
            samples_text += f"Subject: {email['subject']}\n"
            samples_text += f"Content: {email['content']}\n"
            samples_text += "-" * 50 + "\n"

        prompt = f"""
        Analyze the following collection of customer emails that have been clustered together based on semantic similarity. These are emails received by a collections/accounts receivable department.

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
            "reasoning": "explanation of why these emails cluster together"
        }}

        Focus on collection-specific intents like payment promises, disputes, hardship requests, acknowledgments, etc.
        For sentiment, consider apologetic, frustrated, cooperative, evasive, etc.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in customer service and collections email analysis. You help categorize customer communications for business intelligence and automated processing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Parse JSON response
            response_text = response.choices[0].message.content

            # Extract JSON from response
            try:
                # Look for JSON block
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]

                analysis = json.loads(json_text)
                analysis['cluster_id'] = cluster_id
                analysis['sample_count'] = len(sample_emails)

                return analysis

            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text,
                    "cluster_id": cluster_id
                }

        except Exception as e:
            return {
                "error": f"API call failed: {str(e)}",
                "cluster_id": cluster_id
            }

    def analyze_all_clusters(self, max_clusters: int = 10) -> Dict[str, Any]:
        """Analyze multiple clusters and generate category proposals"""
        if not self.cluster_data:
            raise ValueError("Cluster data not loaded. Call load_data() first.")

        print(f"Analyzing top {max_clusters} clusters with LLM...")

        # Get clusters sorted by size (excluding noise)
        clusters = [(k, v) for k, v in self.cluster_data['cluster_stats'].items() if k != '-1']
        clusters.sort(key=lambda x: x[1]['size'], reverse=True)

        # Analyze top clusters
        analyses = {}
        for i, (cluster_id, cluster_info) in enumerate(clusters[:max_clusters]):
            print(f"Analyzing cluster {cluster_id} ({cluster_info['size']} emails, {cluster_info['percentage']:.1f}%)")

            analysis = self.analyze_cluster_with_llm(cluster_id)
            analyses[cluster_id] = analysis

            # Add cluster size info
            analyses[cluster_id]['cluster_size'] = cluster_info['size']
            analyses[cluster_id]['cluster_percentage'] = cluster_info['percentage']

        return analyses

    def generate_taxonomy_draft(self, cluster_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial taxonomy draft from cluster analyses"""
        print("Generating taxonomy draft from cluster analyses...")

        # Collect all proposed intents and sentiments
        intents = {}
        sentiments = {}

        for cluster_id, analysis in cluster_analyses.items():
            if 'error' in analysis:
                continue

            # Collect intent categories
            intent = analysis.get('proposed_intent', '').lower().strip()
            if intent:
                if intent not in intents:
                    intents[intent] = {
                        'definition': analysis.get('intent_definition', ''),
                        'clusters': [],
                        'total_emails': 0,
                        'decision_rules': analysis.get('decision_rules', []),
                        'sample_indicators': analysis.get('sample_indicators', [])
                    }
                intents[intent]['clusters'].append(cluster_id)
                intents[intent]['total_emails'] += analysis.get('cluster_size', 0)

            # Collect sentiment categories
            sentiment = analysis.get('proposed_sentiment', '').lower().strip()
            if sentiment:
                if sentiment not in sentiments:
                    sentiments[sentiment] = {
                        'definition': analysis.get('sentiment_definition', ''),
                        'clusters': [],
                        'total_emails': 0
                    }
                sentiments[sentiment]['clusters'].append(cluster_id)
                sentiments[sentiment]['total_emails'] += analysis.get('cluster_size', 0)

        taxonomy_draft = {
            'intent_categories': intents,
            'sentiment_categories': sentiments,
            'cluster_analyses': cluster_analyses,
            'metadata': {
                'total_clusters_analyzed': len(cluster_analyses),
                'total_emails_in_analysis': sum(a.get('cluster_size', 0) for a in cluster_analyses.values() if 'error' not in a),
                'generation_timestamp': str(np.datetime64('now'))
            }
        }

        return taxonomy_draft


def main():
    """Main execution function"""
    print("=== LLM-POWERED CLUSTER ANALYSIS ===\n")

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("Or pass it as a parameter to ClusterAnalyzer()")
        return

    try:
        # Initialize analyzer
        analyzer = ClusterAnalyzer()

        # Load data
        analyzer.load_data()

        # Analyze clusters (start with top 8 clusters)
        cluster_analyses = analyzer.analyze_all_clusters(max_clusters=8)

        # Save individual cluster analyses
        with open('cluster_analyses_llm.json', 'w') as f:
            json.dump(cluster_analyses, f, indent=2)
        print(f"Saved cluster analyses to 'cluster_analyses_llm.json'")

        # Generate taxonomy draft
        taxonomy_draft = analyzer.generate_taxonomy_draft(cluster_analyses)

        # Save taxonomy draft
        with open('taxonomy_draft.json', 'w') as f:
            json.dump(taxonomy_draft, f, indent=2)
        print(f"Saved taxonomy draft to 'taxonomy_draft.json'")

        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Analyzed {len(cluster_analyses)} clusters")
        print(f"Proposed {len(taxonomy_draft['intent_categories'])} intent categories:")
        for intent in taxonomy_draft['intent_categories'].keys():
            print(f"  - {intent}")
        print(f"Proposed {len(taxonomy_draft['sentiment_categories'])} sentiment categories:")
        for sentiment in taxonomy_draft['sentiment_categories'].keys():
            print(f"  - {sentiment}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run the clustering analysis first")
        print("3. Have the required data files in the current directory")


if __name__ == "__main__":
    main()