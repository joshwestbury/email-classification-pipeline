#!/usr/bin/env python3
"""
Embedding generation module for email taxonomy pipeline.

Generates vector embeddings for emails using sentence transformers.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings for email content using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", include_thread_context: bool = True):
        self.model_name = model_name
        self.include_thread_context = include_thread_context
        self.model = None

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        return self.model

    def extract_incoming_emails(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ONLY incoming emails for individual classification."""
        incoming_emails = []
        total_emails = 0

        emails = data.get('emails', [])
        for idx, email in enumerate(emails):
            total_emails += 1
            if email.get('direction') == 'incoming':
                email_record = {
                    'email_id': email.get('id', f"email_{idx}"),
                    'thread_id': email.get('thread_id', email.get('id', f"thread_{idx}")),
                    'subject': email.get('subject', ''),
                    'content': email.get('content', email.get('message', '')),
                    'direction': email.get('direction'),
                    'sender': email.get('sender', ''),
                    'timestamp': email.get('timestamp', '')
                }
                incoming_emails.append(email_record)

        logger.info(f"Filtered to {len(incoming_emails)} incoming emails from {total_emails} total emails for customer taxonomy analysis")
        return incoming_emails

    def extract_thread_contexts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract thread contexts for contextual analysis."""
        if not self.include_thread_context:
            return []

        # Group emails by thread_id - but only include threads with incoming emails
        threads = {}
        emails = data.get('emails', [])

        for email in emails:
            thread_id = email.get('thread_id', email.get('id'))
            if thread_id not in threads:
                threads[thread_id] = []
            threads[thread_id].append(email)

        thread_contexts = []
        for thread_id, thread_emails in threads.items():
            # Only include threads that contain incoming emails (customer communications)
            incoming_count = sum(1 for e in thread_emails if e.get('direction') == 'incoming')

            if incoming_count > 0:  # Only threads with customer emails
                # Get thread subject (from first email)
                thread_subject = thread_emails[0].get('subject', '') if thread_emails else ''

                # Combine all emails in thread
                thread_text = f"Subject: {thread_subject}\n\n"

                for email_idx, email in enumerate(thread_emails):
                    direction_label = "CUSTOMER" if email.get('direction') == 'incoming' else "COMPANY"
                    thread_text += f"--- Email {email_idx + 1} ({direction_label}) ---\n"
                    content = email.get('content', email.get('message', ''))
                    thread_text += f"{content}\n\n"

                thread_record = {
                    'thread_id': thread_id,
                    'thread_subject': thread_subject,
                    'email_count': len(thread_emails),
                    'thread_content': thread_text,
                    'incoming_count': incoming_count,
                    'outgoing_count': len(thread_emails) - incoming_count
                }
                thread_contexts.append(thread_record)

        logger.info(f"Extracted {len(thread_contexts)} thread contexts for embedding")
        return thread_contexts

    def generate_individual_embeddings(self, incoming_emails: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate embeddings for individual incoming emails."""
        model = self._load_model()

        # Prepare texts for embedding
        texts = []
        metadata = []

        for email in tqdm(incoming_emails, desc="Preparing individual emails"):
            # Combine subject and content for embedding
            subject = email.get('subject', '')
            content = email.get('content', '')

            if subject:
                text = f"Subject: {subject}\n\n{content}"
            else:
                text = content

            texts.append(text)

            # Store metadata for each email
            email_metadata = {
                'email_id': email.get('email_id'),
                'thread_id': email.get('thread_id'),
                'subject': subject,
                'content_length': len(content),
                'sender': email.get('sender', ''),
                'timestamp': email.get('timestamp', '')
            }
            metadata.append(email_metadata)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} individual emails...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings, metadata

    def generate_thread_embeddings(self, thread_contexts: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate embeddings for thread contexts."""
        if not thread_contexts:
            return np.array([]), []

        model = self._load_model()

        # Prepare thread texts for embedding
        texts = []
        metadata = []

        for thread in tqdm(thread_contexts, desc="Preparing thread contexts"):
            texts.append(thread['thread_content'])

            thread_metadata = {
                'thread_id': thread['thread_id'],
                'thread_subject': thread['thread_subject'],
                'email_count': thread['email_count'],
                'incoming_count': thread['incoming_count'],
                'outgoing_count': thread['outgoing_count'],
                'content_length': len(thread['thread_content'])
            }
            metadata.append(thread_metadata)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} thread contexts...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        logger.info(f"Generated thread embeddings shape: {embeddings.shape}")
        return embeddings, metadata

    def generate_embeddings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all embeddings for the dataset."""
        logger.info("Starting embedding generation...")

        # Extract emails and threads
        incoming_emails = self.extract_incoming_emails(data)
        thread_contexts = self.extract_thread_contexts(data)

        # Generate individual email embeddings
        individual_embeddings, individual_metadata = self.generate_individual_embeddings(incoming_emails)

        # Generate thread context embeddings
        thread_embeddings, thread_metadata = self.generate_thread_embeddings(thread_contexts)

        embeddings_data = {
            'individual': {
                'embeddings': individual_embeddings,
                'metadata': individual_metadata,
                'count': len(individual_metadata)
            },
            'thread_context': {
                'embeddings': thread_embeddings,
                'metadata': thread_metadata,
                'count': len(thread_metadata)
            },
            'config': {
                'model_name': self.model_name,
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'include_thread_context': self.include_thread_context
            }
        }

        logger.info("Embedding generation complete")
        return embeddings_data

    def save_embeddings(self, embeddings_data: Dict[str, Any], output_dir: Path) -> None:
        """Save embeddings and metadata to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual email embeddings
        individual_embeddings = embeddings_data['individual']['embeddings']
        individual_metadata = embeddings_data['individual']['metadata']

        np.save(output_dir / 'individual_embeddings.npy', individual_embeddings)
        with open(output_dir / 'individual_metadata.json', 'w') as f:
            json.dump(individual_metadata, f, indent=2)

        logger.info(f"Saved individual embeddings: {individual_embeddings.shape}")

        # Save thread context embeddings if they exist
        thread_embeddings = embeddings_data['thread_context']['embeddings']
        thread_metadata = embeddings_data['thread_context']['metadata']

        if len(thread_embeddings) > 0:
            np.save(output_dir / 'thread_context_embeddings.npy', thread_embeddings)
            with open(output_dir / 'thread_context_metadata.json', 'w') as f:
                json.dump(thread_metadata, f, indent=2)

            logger.info(f"Saved thread context embeddings: {thread_embeddings.shape}")

        # Save configuration
        with open(output_dir / 'embeddings_config.json', 'w') as f:
            json.dump(embeddings_data['config'], f, indent=2)

        logger.info(f"Saved embeddings to {output_dir}")