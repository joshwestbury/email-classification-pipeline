#!/usr/bin/env python3
"""
Collection-specific sentiment analysis module.

Implements pattern-based sentiment detection for customer collection emails
with confidence scoring and business-relevant categories.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: str
    confidence: float
    indicators_found: List[str]
    raw_score: Dict[str, float]


class SentimentAnalyzer:
    """Collection-specific sentiment analysis with pattern matching."""

    def __init__(self, mode: str = "features"):
        """Initialize sentiment patterns and weights.

        Args:
            mode: "features" (default) returns numeric features, "labels" returns sentiment labels
        """
        self.mode = mode
        self.sentiment_patterns = {
            'frustrated': {
                'patterns': [
                    r'\b(?:unacceptable|ridiculous|fed up|sick of|tired of|enough)\b',
                    r'\b(?:still waiting|three times|multiple times|repeatedly)\b',
                    r'\b(?:urgent|immediately|asap|right now|deadline)\b',
                    r'\b(?:frustrated|annoyed|disappointed|upset)\b',
                    r'\b(?:escalat\w*|complain\w*|report\w*)\b',
                    r'\b(?:demand|require|must have|need now)\b',
                    r'\b(?:this is|it is) (?:ridiculous|unacceptable|outrageous)\b'
                ],
                'weight': 1.0,
                'threshold': 0.3
            },
            'cooperative': {
                'patterns': [
                    r'\b(?:working with you|happy to|glad to|pleased to)\b',
                    r'\b(?:trying to resolve|working on|looking into)\b',
                    r'\b(?:understand|appreciate|thank you|thanks)\b',
                    r'\b(?:cooperation|collaborate|partner)\b',
                    r'\b(?:help\w*|assist\w*|support\w*)\b',
                    r'\b(?:best regards|sincerely|kind regards)\b'
                ],
                'weight': 1.0,
                'threshold': 0.25
            },
            'apologetic': {
                'patterns': [
                    r'\b(?:apologize|sorry|regret|my fault)\b',
                    r'\b(?:oversight|mistake|error|delay)\b',
                    r'\b(?:should have|meant to|intended to)\b',
                    r'\b(?:excuse|pardon|forgive)\b',
                    r'\b(?:take responsibility|our fault)\b'
                ],
                'weight': 1.0,
                'threshold': 0.2
            },
            'urgent': {
                'patterns': [
                    r'\b(?:urgent|asap|immediately|right away)\b',
                    r'\b(?:time sensitive|deadline|overdue)\b',
                    r'\b(?:need (?:immediately|now|asap))\b',
                    r'\b(?:critical|emergency|important)\b',
                    r'\b(?:by (?:today|tomorrow|end of day))\b'
                ],
                'weight': 1.0,
                'threshold': 0.25
            },
            'confused': {
                'patterns': [
                    r'\b(?:don\'t understand|unclear|confus\w*)\b',
                    r'\b(?:please explain|help me understand)\b',
                    r'\b(?:not sure|uncertain|unsure)\b',
                    r'\b(?:what (?:do you mean|does this mean))\b',
                    r'\b(?:clarification|clarify)\b'
                ],
                'weight': 1.0,
                'threshold': 0.2
            },
            'professional': {
                'patterns': [
                    r'\b(?:regards|sincerely|best)\b',
                    r'\b(?:please|thank you|thanks)\b',
                    r'\b(?:update|inform|notification)\b',
                    r'\b(?:per|according to|as requested)\b',
                    r'\b(?:attached|enclosed|included)\b'
                ],
                'weight': 0.5,  # Lower weight as it's the default
                'threshold': 0.1
            }
        }

    def _extract_text_for_analysis(self, email_data: Dict[str, Any]) -> str:
        """Extract relevant text content from email for sentiment analysis."""
        text_parts = []

        # Extract subject if available
        if 'subject' in email_data and email_data['subject']:
            text_parts.append(email_data['subject'])

        # Extract main content
        if 'content' in email_data and email_data['content']:
            text_parts.append(email_data['content'])
        elif 'body' in email_data and email_data['body']:
            text_parts.append(email_data['body'])
        elif 'text' in email_data and email_data['text']:
            text_parts.append(email_data['text'])

        return ' '.join(text_parts).lower()

    def _calculate_sentiment_scores(self, text: str) -> Dict[str, Tuple[float, List[str]]]:
        """Calculate sentiment scores for all categories."""
        scores = {}

        for sentiment, config in self.sentiment_patterns.items():
            total_score = 0.0
            found_indicators = []

            for pattern in config['patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Score based on number of matches and pattern weight
                    pattern_score = len(matches) * config['weight']
                    total_score += pattern_score
                    found_indicators.extend(matches)

            # Normalize score by text length to avoid bias toward longer emails
            text_length = max(len(text.split()), 1)  # Avoid division by zero
            normalized_score = total_score / (text_length / 100)  # Per 100 words

            scores[sentiment] = (normalized_score, found_indicators)

        return scores

    def _compute_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Compute linguistic features without category labels.

        Returns:
            Dictionary with feature_summary (numeric scores) and top_feature_examples (text snippets)
        """
        # Calculate raw sentiment scores
        sentiment_scores = self._calculate_sentiment_scores(text)

        # Map sentiment categories to feature names (without revealing category labels)
        feature_mapping = {
            'frustrated': 'escalation_cues',
            'cooperative': 'politeness_markers',
            'apologetic': 'apology_markers',
            'urgent': 'urgency_cues',
            'confused': 'clarification_seeking',
            'professional': 'formality_markers'
        }

        # Additional text-level features
        exclamation_count = text.count('!')
        all_caps_words = len([w for w in text.split() if w.isupper() and len(w) > 1])
        word_count = max(len(text.split()), 1)

        # Build feature summary
        feature_summary = {}
        top_feature_examples = {}

        for sentiment, feature_name in feature_mapping.items():
            score, indicators = sentiment_scores.get(sentiment, (0.0, []))
            feature_summary[feature_name] = round(score, 3)

            # Collect top examples for features with non-zero scores
            if score > 0 and indicators:
                top_feature_examples[feature_name] = indicators[:3]  # Top 3 examples

        # Add computed text features
        feature_summary['exclamation_rate'] = round(exclamation_count / word_count, 3)
        feature_summary['all_caps_rate'] = round(all_caps_words / word_count, 3)

        # Compute gratitude markers (subset of politeness)
        gratitude_pattern = r'\b(?:thank you|thanks|appreciate|grateful)\b'
        gratitude_matches = re.findall(gratitude_pattern, text, re.IGNORECASE)
        feature_summary['gratitude_markers'] = round(len(gratitude_matches) / (word_count / 100), 3)

        # Compute negative lexicon rate
        negative_pattern = r'\b(?:cannot|can\'t|unable|impossible|no|not|never|problem|issue|error|fail)\b'
        negative_matches = re.findall(negative_pattern, text, re.IGNORECASE)
        feature_summary['negative_lexicon_rate'] = round(len(negative_matches) / (word_count / 100), 3)

        if gratitude_matches:
            top_feature_examples['gratitude_markers'] = gratitude_matches[:3]
        if negative_matches:
            top_feature_examples['negative_lexicon_rate'] = negative_matches[:3]

        return {
            'feature_summary': feature_summary,
            'top_feature_examples': top_feature_examples
        }

    def analyze_sentiment(self, email_data: Dict[str, Any]) -> SentimentResult:
        """Analyze sentiment of a single email."""
        text = self._extract_text_for_analysis(email_data)

        if not text.strip():
            return SentimentResult(
                sentiment='professional',
                confidence=0.5,
                indicators_found=[],
                raw_score={'professional': 0.5}
            )

        # Calculate sentiment scores
        sentiment_scores = self._calculate_sentiment_scores(text)

        # Find the highest scoring sentiment above threshold
        best_sentiment = 'professional'
        best_score = 0.0
        best_indicators = []

        for sentiment, (score, indicators) in sentiment_scores.items():
            threshold = self.sentiment_patterns[sentiment]['threshold']
            if score >= threshold and score > best_score:
                best_sentiment = sentiment
                best_score = score
                best_indicators = indicators

        # Calculate confidence based on score margin
        confidence = min(1.0, best_score * 2)  # Scale confidence

        return SentimentResult(
            sentiment=best_sentiment,
            confidence=confidence,
            indicators_found=best_indicators,
            raw_score={k: v[0] for k, v in sentiment_scores.items()}
        )

    def analyze_batch(self, emails: List[Dict[str, Any]]) -> List[SentimentResult]:
        """Analyze sentiment for a batch of emails."""
        logger.info(f"Analyzing sentiment for {len(emails)} emails")

        results = []
        sentiment_counts = {}

        for email in emails:
            result = self.analyze_sentiment(email)
            results.append(result)

            # Track sentiment distribution
            sentiment = result.sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        logger.info(f"Sentiment distribution: {sentiment_counts}")
        return results

    def enrich_clusters_with_sentiment(self, cluster_analysis: Dict[str, Any], emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich cluster analysis with sentiment information.

        Behavior depends on self.mode:
        - "features": Returns linguistic features without category labels
        - "labels": Returns traditional sentiment labels (for debugging)
        """
        logger.info(f"Enriching clusters with sentiment analysis (mode={self.mode})")

        enriched_analysis = cluster_analysis.copy()

        for cluster_id, cluster_info in enriched_analysis.items():
            if cluster_id == '-1':  # Skip noise cluster
                continue

            # Get emails in this cluster
            cluster_emails = []
            if 'sample_indices' in cluster_info:
                for idx in cluster_info['sample_indices']:
                    if idx < len(emails):
                        cluster_emails.append(emails[idx])

            if not cluster_emails:
                continue

            if self.mode == "features":
                # Feature-based analysis (no labels)
                cluster_info['linguistic_preanalysis'] = self._compute_cluster_features(cluster_emails)
            else:
                # Label-based analysis (legacy/debug mode)
                cluster_info['sentiment_analysis'] = self._compute_cluster_labels(cluster_emails, cluster_id)

        logger.info("Cluster sentiment enrichment complete")
        return enriched_analysis

    def _compute_cluster_features(self, cluster_emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated linguistic features for a cluster without labels."""
        all_features = []
        all_examples = {}

        for email in cluster_emails:
            text = self._extract_text_for_analysis(email)
            if not text.strip():
                continue

            features = self._compute_linguistic_features(text)
            all_features.append(features['feature_summary'])

            # Aggregate examples
            for feature_name, examples in features['top_feature_examples'].items():
                if feature_name not in all_examples:
                    all_examples[feature_name] = []
                all_examples[feature_name].extend(examples)

        if not all_features:
            return {'feature_summary': {}, 'top_feature_examples': {}}

        # Average feature scores across cluster
        avg_features = {}
        feature_names = all_features[0].keys()
        for feature_name in feature_names:
            scores = [f[feature_name] for f in all_features if feature_name in f]
            avg_features[feature_name] = round(sum(scores) / len(scores), 3) if scores else 0.0

        # Select top examples (deduplicate and limit)
        top_examples = {}
        for feature_name, examples in all_examples.items():
            # Deduplicate while preserving order
            unique_examples = list(dict.fromkeys(examples))
            top_examples[feature_name] = unique_examples[:3]  # Top 3 unique examples

        return {
            'feature_summary': avg_features,
            'top_feature_examples': top_examples
        }

    def _compute_cluster_labels(self, cluster_emails: List[Dict[str, Any]], cluster_id: str) -> Dict[str, Any]:
        """Compute sentiment labels for a cluster (legacy/debug mode)."""
        sentiment_results = self.analyze_batch(cluster_emails)

        # Calculate cluster sentiment distribution with confidence weighting
        sentiment_dist = {}
        confidence_weighted_dist = {}
        total_confidence = 0.0
        high_confidence_sentiments = []
        minority_sentiments = []

        for result in sentiment_results:
            sentiment = result.sentiment
            confidence = result.confidence

            # Raw count distribution
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1

            # Confidence-weighted distribution
            confidence_weighted_dist[sentiment] = confidence_weighted_dist.get(sentiment, 0.0) + confidence

            total_confidence += confidence

            # Track high-confidence detections (especially emotional ones)
            if confidence > 0.6 and sentiment in ['frustrated', 'angry', 'desperate', 'urgent', 'apologetic']:
                high_confidence_sentiments.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'indicators': result.indicators_found
                })

            # Track minority sentiments with reasonable confidence
            if confidence > 0.15 and sentiment != 'professional':
                minority_sentiments.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'count': sentiment_dist[sentiment]
                })

        # Determine the most representative sentiment using enhanced logic
        dominant_by_count = max(sentiment_dist.items(), key=lambda x: x[1])[0] if sentiment_dist else 'professional'
        dominant_by_confidence = max(confidence_weighted_dist.items(), key=lambda x: x[1])[0] if confidence_weighted_dist else 'professional'

        # Priority-based sentiment selection
        selected_sentiment = dominant_by_count

        # Override with high-confidence emotional sentiments (even if minority)
        if high_confidence_sentiments:
            # Sort by confidence and choose the highest
            best_emotional = max(high_confidence_sentiments, key=lambda x: x['confidence'])
            selected_sentiment = best_emotional['sentiment']
            logger.info(f"Cluster {cluster_id}: Overriding with high-confidence minority sentiment '{selected_sentiment}' (confidence: {best_emotional['confidence']:.2f})")

        # If no high-confidence emotional, but strong minority sentiments exist
        elif minority_sentiments:
            # Filter for significant minority sentiments (>15% confidence)
            significant_minorities = [s for s in minority_sentiments if s['confidence'] > 0.25]
            if significant_minorities:
                # Prioritize emotional sentiments over neutral ones
                emotional_priorities = ['frustrated', 'angry', 'desperate', 'urgent', 'apologetic', 'cooperative']
                for priority_sentiment in emotional_priorities:
                    for minority in significant_minorities:
                        if minority['sentiment'] == priority_sentiment:
                            selected_sentiment = priority_sentiment
                            logger.info(f"Cluster {cluster_id}: Selected priority minority sentiment '{selected_sentiment}' over dominant '{dominant_by_count}'")
                            break
                    if selected_sentiment == priority_sentiment:
                        break

        return {
            'distribution': sentiment_dist,
            'confidence_weighted_distribution': confidence_weighted_dist,
            'dominant_sentiment': selected_sentiment,
            'dominant_by_count': dominant_by_count,
            'dominant_by_confidence': dominant_by_confidence,
            'avg_confidence': total_confidence / len(sentiment_results) if sentiment_results else 0.0,
            'high_confidence_sentiments': high_confidence_sentiments,
            'minority_sentiments': minority_sentiments,
            'sample_results': [
                {
                    'sentiment': r.sentiment,
                    'confidence': r.confidence,
                    'indicators': r.indicators_found[:3]  # Top 3 indicators
                }
                for r in sentiment_results[:5]  # Top 5 samples
            ]
        }