"""
OpenAI embedding generation module.

This module provides functionality to generate embeddings using OpenAI's
text-embedding models, specifically optimized for the Ada embedding endpoint.
"""

import openai
import pandas as pd
import numpy as np
import time
import os
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from ..utils.config_loader import config
from ..utils.logging_utils import setup_logger
from ..utils.file_io import save_dataframe, load_dataframe

logger = setup_logger(__name__)


class OpenAIEmbedder:
    """
    Handles OpenAI embedding generation using the text-embedding models.
    
    This class provides methods to generate high-quality embeddings using OpenAI's
    embedding API, with support for batch processing and progress tracking.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the OpenAI embedder.
        
        Args:
            api_key: OpenAI API key. If None, expects OPENAI_API_KEY environment variable.
            model_name: Name of the embedding model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        
        api_config = config.get_api_config('openai')
        self.model_name = model_name or api_config.get('embedding_model', 'text-embedding-3-small')
        
        # Determine embedding dimension based on model
        self.embedding_dim = self._get_embedding_dimension()
        
        # Rate limiting parameters
        self.requests_per_minute = 3000  # OpenAI rate limit
        self.min_request_interval = 60 / self.requests_per_minute
        
        logger.info(f"Initialized OpenAIEmbedder with model: {self.model_name} ({self.embedding_dim}d)")
    
    def _get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension for the specified model.
        
        Returns:
            Embedding dimension size
        """
        model_dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        
        return model_dimensions.get(self.model_name, 1536)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or pd.isna(text):
            text = ""
        
        try:
            # Clean text for API
            text = text.replace("\n", " ").strip()
            
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding)
        
        except Exception as e:
            logger.warning(f"Error generating embedding for text: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        save_progress: bool = True,
        progress_file: Optional[str] = None,
        resume_from_checkpoint: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts with progress tracking.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            save_progress: Whether to save progress periodically
            progress_file: Path to save/load progress
            resume_from_checkpoint: Whether to resume from existing progress
            
        Returns:
            2D numpy array of embeddings
        """
        total_texts = len(texts)
        
        if progress_file is None:
            progress_file = Path(config.get('data.embeddings_path')) / "openai_embedding_progress.csv"
        
        # Initialize or load progress
        embeddings = []
        start_idx = 0
        
        if resume_from_checkpoint and save_progress and Path(progress_file).exists():
            try:
                progress_df = load_dataframe(progress_file)
                start_idx = len(progress_df)
                embeddings = [np.array(row) for row in progress_df.filter(regex='^embed_').values]
                logger.info(f"Resuming from checkpoint at index {start_idx}")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}. Starting from beginning.")
                start_idx = 0
                embeddings = []
        
        logger.info(f"Starting OpenAI embedding generation for {total_texts} texts (from index {start_idx})")
        
        for i in range(start_idx, total_texts):
            try:
                # Rate limiting
                start_time = time.time()
                
                # Generate embedding
                text = texts[i]
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                
                # Log progress
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / total_texts * 100
                    logger.info(f"Processed {i + 1}/{total_texts} texts ({progress:.1f}%)")
                
                # Save progress periodically
                if save_progress and (i + 1) % batch_size == 0:
                    self._save_embedding_progress(texts[:i+1], embeddings, progress_file)
                    logger.info(f"Saved progress after processing {i + 1} texts")
                
                # Rate limiting delay
                elapsed = time.time() - start_time
                if elapsed < self.min_request_interval:
                    time.sleep(self.min_request_interval - elapsed)
            
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                embeddings.append(np.zeros(self.embedding_dim))
        
        # Final save
        if save_progress:
            self._save_embedding_progress(texts, embeddings, progress_file)
        
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Generated OpenAI embeddings: {embeddings_array.shape}")
        
        return embeddings_array
    
    def _save_embedding_progress(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        progress_file: str
    ) -> None:
        """
        Save embedding progress to file.
        
        Args:
            texts: List of processed texts
            embeddings: List of corresponding embeddings
            progress_file: Path to save progress
        """
        try:
            # Create DataFrame with texts and embeddings
            data = {'text': texts[:len(embeddings)]}
            
            # Add embedding columns
            if embeddings:
                embedding_df = pd.DataFrame(
                    np.vstack(embeddings),
                    columns=[f'embed_{i}' for i in range(self.embedding_dim)]
                )
                
                progress_df = pd.concat([
                    pd.DataFrame(data),
                    embedding_df
                ], axis=1)
            else:
                progress_df = pd.DataFrame(data)
            
            save_dataframe(progress_df, progress_file)
        
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        output_prefix: str = 'openai',
        save_path: Optional[str] = None,
        resume_from_checkpoint: bool = True
    ) -> pd.DataFrame:
        """
        Generate OpenAI embeddings for texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_prefix: Prefix for embedding column names
            save_path: Optional path to save the result
            resume_from_checkpoint: Whether to resume from existing progress
            
        Returns:
            DataFrame with added embedding columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Generate embeddings
        embeddings = self.embed_batch(
            texts=df[text_column].fillna('').tolist(),
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Create embedding DataFrame
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'{output_prefix}_embed_{i}' for i in range(self.embedding_dim)],
            index=df.index
        )
        
        # Combine with original DataFrame
        result_df = pd.concat([df, embedding_df], axis=1)
        
        if save_path:
            save_dataframe(result_df, save_path)
            logger.info(f"Saved OpenAI embeddings to {save_path}")
        
        return result_df
    
    def estimate_cost(self, texts: List[str]) -> Dict[str, float]:
        """
        Estimate the cost of embedding a list of texts.
        
        Args:
            texts: List of texts to estimate cost for
            
        Returns:
            Dictionary with cost estimation details
        """
        # Rough token estimation (1 token ≈ 0.75 words)
        total_words = sum(len(str(text).split()) for text in texts)
        estimated_tokens = total_words / 0.75
        
        # Cost per 1000 tokens (as of 2024 - may change)
        cost_per_1k_tokens = {
            'text-embedding-3-small': 0.00002,
            'text-embedding-3-large': 0.00013,
            'text-embedding-ada-002': 0.0001
        }
        
        rate = cost_per_1k_tokens.get(self.model_name, 0.0001)
        estimated_cost = (estimated_tokens / 1000) * rate
        
        return {
            'num_texts': len(texts),
            'estimated_tokens': int(estimated_tokens),
            'cost_per_1k_tokens': rate,
            'estimated_cost_usd': estimated_cost,
            'model': self.model_name
        }
    
    def compare_with_alternatives(
        self,
        texts: List[str],
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """
        Compare OpenAI embeddings with alternative models on a sample.
        
        Args:
            texts: List of texts to analyze
            sample_size: Number of texts to sample for comparison
            
        Returns:
            Dictionary with comparison statistics
        """
        # This would require implementing other embedding methods for comparison
        # For now, we'll just return basic statistics about the OpenAI embeddings
        
        if len(texts) > sample_size:
            sample_texts = np.random.choice(texts, sample_size, replace=False).tolist()
        else:
            sample_texts = texts
        
        logger.info(f"Analyzing OpenAI embeddings on {len(sample_texts)} texts")
        
        # Generate embeddings
        embeddings = self.embed_batch(sample_texts, save_progress=False)
        
        # Calculate statistics
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Calculate pairwise similarities for diversity analysis
        similarities = []
        for i in range(min(10, len(embeddings))):  # Sample 10 pairs to avoid O(n²)
            for j in range(i+1, min(10, len(embeddings))):
                similarity = np.dot(embeddings[i], embeddings[j]) / (norms[i] * norms[j])
                similarities.append(similarity)
        
        analysis = {
            'sample_size': len(sample_texts),
            'embedding_dimension': self.embedding_dim,
            'avg_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'avg_pairwise_similarity': np.mean(similarities) if similarities else 0,
            'std_pairwise_similarity': np.std(similarities) if similarities else 0,
            'model': self.model_name
        }
        
        logger.info(f"OpenAI embedding analysis: avg norm: {analysis['avg_norm']:.3f}, "
                   f"avg similarity: {analysis['avg_pairwise_similarity']:.3f}")
        
        return analysis


def main():
    """Example usage of the OpenAIEmbedder class."""
    input_path = Path(config.get('data.processed_data_path')) / "scrubbed_posts.csv"
    output_path = Path(config.get('data.embeddings_path')) / "openai_embeddings.csv"
    
    try:
        # Load data
        df = load_dataframe(input_path)
        logger.info(f"Loaded {len(df)} texts for OpenAI embedding generation")
        
        # Initialize embedder
        embedder = OpenAIEmbedder()
        
        # Estimate cost
        cost_estimate = embedder.estimate_cost(df['scrubbed_text'].fillna('').tolist())
        logger.info(f"Cost estimate: {cost_estimate}")
        
        # Generate embeddings (using a sample for demo)
        sample_df = df.head(100)  # Process first 100 for demo
        result_df = embedder.embed_dataframe(
            df=sample_df,
            text_column='scrubbed_text',
            output_prefix='openai_ada',
            save_path=output_path
        )
        
        # Analyze embeddings
        analysis = embedder.compare_with_alternatives(
            sample_df['scrubbed_text'].fillna('').tolist()
        )
        logger.info(f"Embedding analysis: {analysis}")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()