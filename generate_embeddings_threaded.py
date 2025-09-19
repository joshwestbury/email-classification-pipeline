#!/usr/bin/env python3
"""
Generate embeddings for master email threads - Option A approach.

This script generates two types of embeddings:
1. Individual incoming email embeddings (primary classification target)
2. Thread context embeddings (for contextual analysis)

Uses the anonymized master email threads dataset.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_master_email_threads(filepath: str) -> Dict[str, Any]:
    """Load master email threads data from JSON file."""
    logger.info(f"Loading master email threads from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_threads = len(data['email_threads'])
    total_emails = data['metadata']['final_total_emails']
    incoming_emails = data['metadata']['final_incoming_emails']

    logger.info(f"Loaded {total_threads} threads with {total_emails} total emails")
    logger.info(f"Target incoming emails for classification: {incoming_emails}")

    return data


def extract_incoming_emails(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all incoming emails for individual classification."""
    incoming_emails = []

    for thread in data['email_threads']:
        thread_id = thread['original_id']
        thread_subject = thread['subject']

        for email_idx, email in enumerate(thread['emails']):
            if email['direction'] == 'incoming':
                # Add thread context to each incoming email
                email_record = {
                    'thread_id': thread_id,
                    'thread_subject': thread_subject,
                    'email_index': email_idx,
                    'content': email['content'],
                    'direction': email['direction'],
                    'type': email['type'],
                    'original_content_length': email.get('original_content_length', len(email['content'])),
                    'signature_info': email.get('signature_info', {}),
                    'content_anonymized': email.get('content_anonymized', False)
                }
                incoming_emails.append(email_record)

    logger.info(f"Extracted {len(incoming_emails)} incoming emails")
    return incoming_emails


def extract_thread_contexts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract thread contexts (full conversations) for contextual analysis."""
    thread_contexts = []

    for thread in data['email_threads']:
        if thread['email_count'] > 1:  # Only threads with multiple emails
            # Check if thread has incoming emails
            has_incoming = any(email['direction'] == 'incoming' for email in thread['emails'])

            if has_incoming:
                # Combine all emails in thread chronologically
                thread_text = f"Subject: {thread['subject']}\n\n"

                for email_idx, email in enumerate(thread['emails']):
                    direction_label = "CUSTOMER" if email['direction'] == 'incoming' else "LITERA"
                    thread_text += f"--- Email {email_idx + 1} ({direction_label}) ---\n"
                    thread_text += f"{email['content']}\n\n"

                thread_record = {
                    'thread_id': thread['original_id'],
                    'thread_subject': thread['subject'],
                    'email_count': thread['email_count'],
                    'thread_content': thread_text,
                    'incoming_count': sum(1 for e in thread['emails'] if e['direction'] == 'incoming'),
                    'outgoing_count': sum(1 for e in thread['emails'] if e['direction'] == 'outgoing')
                }
                thread_contexts.append(thread_record)

    logger.info(f"Extracted {len(thread_contexts)} thread contexts with incoming emails")
    return thread_contexts


def extract_text_for_embedding(email: Dict[str, Any], context_type: str = "individual") -> str:
    """
    Extract and prepare text content for embedding generation.

    Args:
        email: Email record dictionary
        context_type: "individual" for single emails, "thread" for full contexts
    """
    if context_type == "individual":
        # For individual emails, use subject context + content
        thread_subject = email.get('thread_subject', '')
        content = email.get('content', '')

        # Combine thread subject and email content
        combined_text = f"Thread: {thread_subject}\n\nCustomer Email: {content}"

    elif context_type == "thread":
        # For thread contexts, use the full conversation
        combined_text = email.get('thread_content', '')

    return combined_text.strip()


def generate_embeddings(
    texts: List[str],
    description: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Generate embeddings for texts using sentence transformer model.
    """
    logger.info(f"Initializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Generate embeddings with progress bar
    logger.info(f"Generating embeddings for {description}...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # Normalize for better clustering performance
    )

    logger.info(f"Generated {description} embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings_and_metadata(
    incoming_embeddings: np.ndarray,
    incoming_emails: List[Dict[str, Any]],
    thread_embeddings: np.ndarray,
    thread_contexts: List[Dict[str, Any]],
    output_dir: str = "."
) -> None:
    """Save embeddings and associated metadata to files."""
    output_path = Path(output_dir)

    # Save incoming email embeddings
    incoming_embeddings_file = output_path / "incoming_email_embeddings.npy"
    logger.info(f"Saving incoming email embeddings to {incoming_embeddings_file}")
    np.save(incoming_embeddings_file, incoming_embeddings)

    # Save thread context embeddings
    thread_embeddings_file = output_path / "thread_context_embeddings.npy"
    logger.info(f"Saving thread context embeddings to {thread_embeddings_file}")
    np.save(thread_embeddings_file, thread_embeddings)

    # Save incoming email metadata
    incoming_metadata = [
        {
            "index": idx,
            "thread_id": email["thread_id"],
            "thread_subject": email["thread_subject"],
            "email_index": email["email_index"],
            "content_length": len(email["content"]),
            "original_content_length": email["original_content_length"],
            "content_anonymized": email["content_anonymized"],
            "has_signature": bool(email.get("signature_info", {}))
        }
        for idx, email in enumerate(incoming_emails)
    ]

    incoming_metadata_file = output_path / "incoming_email_metadata.json"
    logger.info(f"Saving incoming email metadata to {incoming_metadata_file}")
    with open(incoming_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(incoming_metadata, f, indent=2, ensure_ascii=False)

    # Save thread context metadata
    thread_metadata = [
        {
            "index": idx,
            "thread_id": thread["thread_id"],
            "thread_subject": thread["thread_subject"],
            "email_count": thread["email_count"],
            "incoming_count": thread["incoming_count"],
            "outgoing_count": thread["outgoing_count"],
            "thread_content_length": len(thread["thread_content"])
        }
        for idx, thread in enumerate(thread_contexts)
    ]

    thread_metadata_file = output_path / "thread_context_metadata.json"
    logger.info(f"Saving thread context metadata to {thread_metadata_file}")
    with open(thread_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(thread_metadata, f, indent=2, ensure_ascii=False)

    # Save embedding configuration for reproducibility
    config = {
        "model_name": "all-MiniLM-L6-v2",
        "approach": "Option A - Separate individual and thread embeddings",
        "incoming_emails": {
            "count": len(incoming_emails),
            "embedding_dimension": incoming_embeddings.shape[1],
            "description": "Individual incoming emails for primary classification"
        },
        "thread_contexts": {
            "count": len(thread_contexts),
            "embedding_dimension": thread_embeddings.shape[1],
            "description": "Full thread contexts for contextual analysis"
        },
        "normalization": "L2 normalized",
        "batch_size": 32,
        "text_preprocessing": "Thread subject + customer email content for individuals, full conversation for threads"
    }

    config_file = output_path / "embeddings_config.json"
    logger.info(f"Saving configuration to {config_file}")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def main():
    """Main function to generate embeddings for master email threads."""
    # File paths
    input_file = "master_email_threads_anonymized.json"

    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found!")
        return

    try:
        # Load master email threads data
        data = load_master_email_threads(input_file)

        # Extract incoming emails for individual classification
        incoming_emails = extract_incoming_emails(data)

        # Extract thread contexts for contextual analysis
        thread_contexts = extract_thread_contexts(data)

        # Prepare texts for embedding
        logger.info("Preparing texts for embedding generation...")

        incoming_texts = [
            extract_text_for_embedding(email, "individual")
            for email in incoming_emails
        ]

        thread_texts = [
            extract_text_for_embedding(thread, "thread")
            for thread in thread_contexts
        ]

        # Generate embeddings for both types
        incoming_embeddings = generate_embeddings(
            incoming_texts,
            "incoming emails (individual classification)"
        )

        thread_embeddings = generate_embeddings(
            thread_texts,
            "thread contexts (contextual analysis)"
        )

        # Save results
        save_embeddings_and_metadata(
            incoming_embeddings, incoming_emails,
            thread_embeddings, thread_contexts
        )

        logger.info("Embedding generation completed successfully!")
        logger.info(f"Generated embeddings:")
        logger.info(f"  - {incoming_embeddings.shape[0]} incoming email embeddings ({incoming_embeddings.shape[1]} dims)")
        logger.info(f"  - {thread_embeddings.shape[0]} thread context embeddings ({thread_embeddings.shape[1]} dims)")
        logger.info("Files created:")
        logger.info("  - incoming_email_embeddings.npy")
        logger.info("  - thread_context_embeddings.npy")
        logger.info("  - incoming_email_metadata.json")
        logger.info("  - thread_context_metadata.json")
        logger.info("  - embeddings_config.json")

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()