#!/usr/bin/env python3
"""
Test the improved rich curation functionality
"""

import json
from pathlib import Path
from pipeline.curator import TaxonomyCurator

def test_rich_curation():
    """Test the rich curation with existing analysis."""

    # Load existing analysis
    analysis_file = "outputs/collection_notes_test/taxonomy_analysis.json"
    if not Path(analysis_file).exists():
        print(f"Analysis file {analysis_file} not found")
        return

    with open(analysis_file, 'r') as f:
        llm_analysis = json.load(f)

    print("Loaded LLM analysis with:")
    print(f"  - {len(llm_analysis['cluster_analyses'])} cluster analyses")
    print(f"  - {llm_analysis['analysis_summary']['total_emails_in_analyzed_clusters']} emails analyzed")

    # Test rich curation
    curator = TaxonomyCurator()

    print("\n=== Testing Rich Curation ===")
    try:
        results = curator.curate_taxonomy(llm_analysis)

        final_taxonomy = results['final_taxonomy']
        curation_stats = results['curation_stats']

        print(f"✅ Rich curation successful!")
        print(f"Intent categories: {curation_stats['final_intent_categories']}")
        print(f"Sentiment categories: {curation_stats['final_sentiment_categories']}")

        # Show structure comparison by checking if taxonomy contains expected sections
        print(f"\n📊 Structure Comparison:")
        print(f"  ✅ Has modifier_flags: {'modifier_flags' in final_taxonomy}")
        print(f"  ✅ Has validation_rules: {'validation_rules' in final_taxonomy}")
        print(f"  ✅ Has netsuite_integration: {'netsuite_integration' in final_taxonomy}")
        print(f"  ✅ Has header comments: {'# Collection Notes AI' in final_taxonomy}")
        print(f"  ✅ Has clean formatting: {'intent_categories:' in final_taxonomy}")

        # Show a sample of the generated YAML
        print(f"\n📝 Sample Generated YAML:")
        lines = final_taxonomy.split('\n')
        for i, line in enumerate(lines[:15]):
            print(f"  {line}")
        print("  ...")

        # Save the improved taxonomy
        output_dir = Path("outputs/collection_notes_rich")
        output_dir.mkdir(parents=True, exist_ok=True)
        curator.save_results(results, output_dir)
        print(f"\n💾 Saved rich taxonomy to {output_dir}")

    except Exception as e:
        print(f"❌ Rich curation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rich_curation()