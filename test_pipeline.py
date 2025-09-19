#!/usr/bin/env python3
"""
Test script to verify the pipeline can handle litera_raw_emails.json
"""

import json
from pathlib import Path
from pipeline.config import PipelineConfig
from pipeline.data_processor import DataProcessor

def test_thread_separation():
    """Test the thread separation functionality with a sample email."""

    # Sample email with threaded content
    sample_email = {
        "id": 3276341,
        "subject": "Re: Support Suspended | 60 Days Past Due - Reminder: Invoice INV166216 from",
        "message": """<div dir="ltr"><div>Hi Joshua, </div><div><br></div><div>Invoice has been GR'd and sent to AP for processing.  This should be included in this week's pay run.  </div><div><br><div class="gmail_quote gmail_quote_container"><div dir="ltr" class="gmail_attr">On Tue, 16 Sept 2025 at 20:12, Joshua Smith &lt;&lt;a href="mailto:joshua.smith@holcim.com"&gt;joshua.smith@holcim.com&lt;/a&gt;&gt; wrote:&lt;br&gt;</div><blockquote class="gmail_quote" style="margin:0px 0px 0px 0.8ex;border-left:1px solid rgb(204,204,204);padding-left:1ex"><div dir="auto"><div>Hi Noreen,</div><div dir="auto"><br></div><div dir="auto">Are you able to confirm when payment will be processed?</div></blockquote></div>"""
    }

    processor = DataProcessor(clean_html=True, separate_threads=True)

    # Test thread separation
    separated = processor.separate_email_threads([sample_email])

    print(f"Original email count: 1")
    print(f"Separated email count: {len(separated)}")

    for i, email in enumerate(separated):
        print(f"\nEmail {i + 1}:")
        print(f"  ID: {email['id']}")
        print(f"  Direction: {email['direction']}")
        print(f"  Thread Position: {email['thread_position']}")
        print(f"  Content: {email['content'][:100]}...")

def test_full_file_processing():
    """Test processing a few records from the actual file."""

    input_file = "Collection Notes - Sentiment Analysis/litera_raw_emails.json"

    if not Path(input_file).exists():
        print(f"File {input_file} not found - skipping full file test")
        return

    processor = DataProcessor(clean_html=True, separate_threads=True)

    # Load just the first 5 records for testing
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        data = json.loads(content)
        sample_data = data[:5]  # First 5 records
    except json.JSONDecodeError:
        print("JSON parsing failed - using malformed JSON parser")
        all_data = processor.parse_malformed_json(content)
        sample_data = all_data[:5]

    print(f"\nTesting with {len(sample_data)} sample emails...")

    # Process the sample
    result = processor.process_emails_from_data(sample_data)

    print(f"Processed {result['metadata']['total_emails']} emails")
    print(f"Incoming: {result['metadata']['incoming_emails']}")
    print(f"Outgoing: {result['metadata']['outgoing_emails']}")

    # Show some sample results
    for i, email in enumerate(result['emails'][:3]):
        print(f"\nProcessed Email {i + 1}:")
        print(f"  ID: {email['id']}")
        print(f"  Direction: {email['direction']}")
        print(f"  Thread Separated: {email.get('is_thread_separated', False)}")
        print(f"  Content: {email['content'][:100]}...")

if __name__ == "__main__":
    print("Testing thread separation functionality...")
    test_thread_separation()

    print("\n" + "="*50)
    print("Testing full file processing...")
    test_full_file_processing()