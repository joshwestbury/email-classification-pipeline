#!/usr/bin/env python3
"""
Generate embeddings for anonymized collection emails.

This script processes the anonymized email dataset and generates vector embeddings
using a pre-trained sentence transformer model optimized for business text.
The embeddings will be used for clustering analysis in Phase 1 of taxonomy discovery.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_anonymized_emails(filepath: str) -> List[Dict[str, Any]]:
    """Load anonymized email data from JSON file."""
    logger.info(f"Loading anonymized emails from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        emails = json.load(f)
    logger.info(f"Loaded {len(emails)} emails")
    return emails


def extract_text_for_embedding(email: Dict[str, Any]) -> str:
    """
    Extract and prepare text content for embedding generation.

    Combines subject and message content, focusing on the anonymized message
    which preserves semantic meaning while protecting PII.
    """
    subject = email.get('subject') or ''
    message = email.get('message') or ''

    # Handle None values and strip whitespace
    subject = str(subject).strip()
    message = str(message).strip()

    # Combine subject and message with clear separation
    combined_text = f"Subject: {subject}\n\nMessage: {message}"

    return combined_text


def generate_embeddings(
    emails: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Generate embeddings for email texts using sentence transformer model.

    Uses all-MiniLM-L6-v2 which is optimized for semantic similarity tasks
    and performs well on business/professional text.
    """
    logger.info(f"Initializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract texts for embedding
    logger.info("Extracting text content for embedding generation")
    texts = [extract_text_for_embedding(email) for email in emails]

    # Generate embeddings with progress bar
    logger.info("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # Normalize for better clustering performance
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    emails: List[Dict[str, Any]],
    output_dir: str = "."
) -> None:
    """Save embeddings and associated metadata to files."""
    output_path = Path(output_dir)

    # Save embeddings as numpy array
    embeddings_file = output_path / "email_embeddings.npy"
    logger.info(f"Saving embeddings to {embeddings_file}")
    np.save(embeddings_file, embeddings)

    # Save metadata (email IDs and basic info for clustering analysis)
    metadata = [
        {
            "id": email["id"],
            "subject": email["subject"],
            "original_subject": email.get("original_subject", ""),
            "message_length": len(email.get("message", "")),
            "has_pii": email.get("pii_anonymized", False)
        }
        for email in emails
    ]

    metadata_file = output_path / "email_metadata.json"
    logger.info(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save embedding configuration for reproducibility
    config = {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dimension": embeddings.shape[1],
        "num_emails": embeddings.shape[0],
        "normalization": "L2 normalized",
        "batch_size": 32,
        "generated_timestamp": str(Path(__file__).stat().st_mtime)
    }

    config_file = output_path / "embedding_config.json"
    logger.info(f"Saving configuration to {config_file}")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def main():
    """Main function to generate embeddings for collection emails."""
    # File paths
    input_file = "emails_anonymized.json"

    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found!")
        return

    try:
        # Load anonymized emails
        emails = load_anonymized_emails(input_file)

        # Generate embeddings
        embeddings = generate_embeddings(emails)

        # Save results
        save_embeddings_and_metadata(embeddings, emails)

        logger.info("Embedding generation completed successfully!")
        logger.info(f"Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
        logger.info("Files created:")
        logger.info("  - email_embeddings.npy (embedding vectors)")
        logger.info("  - email_metadata.json (email metadata)")
        logger.info("  - embedding_config.json (generation configuration)")

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise


if __name__ == "__main__":
    main()