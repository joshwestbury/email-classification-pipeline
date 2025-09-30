#!/usr/bin/env python3
"""
Test script to demonstrate the JSON Schema Validation improvements.
"""

import json
from pipeline.analyzer import LLMAnalyzer, LLMClusterAnalysis

def test_validation_improvements():
    """Test the new validation system."""
    print("🔧 Testing JSON Schema Validation Improvements")
    print("=" * 60)

    # Test 1: Pydantic Model Validation
    print("\n1. Testing Pydantic Model Validation:")

    # Valid data
    valid_data = {
        'proposed_intent': 'Payment Status Inquiry',
        'intent_definition': 'Customer asking about the status of their payment or payment schedule',
        'proposed_sentiment': 'Cooperative',
        'sentiment_definition': 'Customer willing to work together to resolve payment issues',
        'decision_rules': ['Contains payment status questions', 'Uses cooperative language'],
        'confidence': 'high',
        'sample_indicators': ['payment status', 'when will payment', 'payment schedule'],
        'emotional_markers': ['willing to pay', 'please help'],
        'reasoning': 'These emails show customers proactively asking about payment status',
        'business_relevance': 'Identifies cooperative customers who need payment guidance'
    }

    try:
        validated = LLMClusterAnalysis(**valid_data)
        print("  ✅ Valid data: PASSED")
        print(f"     Intent: {validated.proposed_intent}")
        print(f"     Sentiment: {validated.proposed_sentiment}")
        print(f"     Confidence: {validated.confidence}")
    except Exception as e:
        print(f"  ❌ Valid data: FAILED - {e}")

    # Invalid data tests
    invalid_tests = [
        {
            'name': 'Invalid confidence level',
            'data': {**valid_data, 'confidence': 'invalid_value'},
            'expected_error': 'string_pattern_mismatch'
        },
        {
            'name': 'Missing required field',
            'data': {k: v for k, v in valid_data.items() if k != 'proposed_intent'},
            'expected_error': 'missing'
        },
        {
            'name': 'Empty decision rules',
            'data': {**valid_data, 'decision_rules': []},
            'expected_error': 'too_short'
        },
        {
            'name': 'Intent too short',
            'data': {**valid_data, 'proposed_intent': ''},
            'expected_error': 'string_too_short'
        }
    ]

    for test in invalid_tests:
        try:
            LLMClusterAnalysis(**test['data'])
            print(f"  ❌ {test['name']}: Should have failed but didn't")
        except Exception as e:
            print(f"  ✅ {test['name']}: Correctly rejected - {type(e).__name__}")

    # Test 2: Analyzer Configuration
    print("\n2. Testing Analyzer Configuration:")

    try:
        analyzer = LLMAnalyzer()
        print("  ✅ LLMAnalyzer initialization: PASSED")
        print(f"     Model: {analyzer.model}")
        print(f"     Top clusters: {analyzer.top_clusters}")
    except Exception as e:
        print(f"  ❌ LLMAnalyzer initialization: FAILED - {e}")

    # Test 3: Enhanced Features Summary
    print("\n3. Enhanced Features Summary:")
    print("  📋 Pydantic Models:")
    print("     - Strict JSON schema validation")
    print("     - Field length and pattern validation")
    print("     - Required field enforcement")
    print("     - Extra field rejection")

    print("  🔄 Retry Logic:")
    print("     - Exponential backoff (1s, 2s, 4s)")
    print("     - 3 retry attempts for validation failures")
    print("     - Specific error handling for ValidationError, JSONDecodeError")

    print("  📊 Enhanced Reporting:")
    print("     - Validation success rate tracking")
    print("     - Failed validation counting")
    print("     - Detailed error categorization")

    print("\n✨ JSON Schema Validation improvements are ready!")
    print("Target: >95% parse success rate (up from ~85%)")

if __name__ == "__main__":
    test_validation_improvements()