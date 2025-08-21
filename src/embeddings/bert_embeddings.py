"""
BERT embedding generation module.

This module provides functionality to generate embeddings using BERT models,
including both CLS token embeddings and mean pooled embeddings.
"""

import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Optional
import logging
from pathlib import Path

from ..utils.config_loader import config
from ..utils.logging_utils import setup_logger
from ..utils.file_io import save_dataframe, load_dataframe

logger = setup_logger(__name__)


class BertEmbedder:
    """
    Handles BERT embedding generation with support for both CLS and mean pooling strategies.
    
    This class encapsulates BERT model loading and provides methods to generate
    768-dimensional embeddings from text using different pooling strategies.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the BERT embedder.
        
        Args:
            model_name: Name of the BERT model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
        """
        bert_config = config.get_model_config('bert')
        self.model_name = model_name or bert_config.get('model_name', 'bert-base-uncased')
        self.max_length = bert_config.get('max_length', 512)
        self.embedding_dim = bert_config.get('embedding_dim', 768)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"Initialized BertEmbedder with {self.model_name} on {self.device}")
    
    def _load_model(self) -> None:
        """Load BERT model and tokenizer."""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded BERT model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model {self.model_name}: {e}")
            raise
    
    def _prepare_inputs(self, text: str) -> dict:
        """
        Prepare input tensors for BERT model.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary of input tensors
        """
        if not text or pd.isna(text):
            text = ""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move tensors to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        return inputs
    
    def get_cls_embedding(self, text: str) -> np.ndarray:
        """
        Generate CLS token embedding for a single text.
        
        The CLS token embedding is the first token's representation from the last
        hidden layer, often used as a sentence-level representation.
        
        Args:
            text: Input text
            
        Returns:
            768-dimensional numpy array representing the CLS embedding
        """
        try:
            inputs = self._prepare_inputs(text)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Extract CLS token embedding (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            
            return cls_embedding.cpu().numpy()
        
        except Exception as e:
            logger.warning(f"Error generating CLS embedding for text: {e}")
            return np.zeros(self.embedding_dim)
    
    def get_mean_embedding(self, text: str) -> np.ndarray:
        """
        Generate mean pooled embedding for a single text.
        
        Mean pooling averages all token representations from the last hidden layer,
        providing a different sentence-level representation strategy.
        
        Args:
            text: Input text
            
        Returns:
            768-dimensional numpy array representing the mean pooled embedding
        """
        try:
            inputs = self._prepare_inputs(text)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling across sequence length dimension
                mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return mean_embedding.cpu().numpy()
        
        except Exception as e:
            logger.warning(f"Error generating mean embedding for text: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_batch(
        self,
        texts: List[str],
        strategy: str = 'cls',
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            strategy: Embedding strategy ('cls' or 'mean')
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress logs
            
        Returns:
            2D numpy array of shape (n_texts, embedding_dim)
        """
        if strategy not in ['cls', 'mean']:
            raise ValueError("Strategy must be 'cls' or 'mean'")
        
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"Starting {strategy.upper()} embedding generation for {total_texts} texts")
        
        # Choose embedding function
        embed_func = self.get_cls_embedding if strategy == 'cls' else self.get_mean_embedding
        
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = embed_func(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            if show_progress and (i + batch_size) % (batch_size * 10) == 0:
                progress = min(i + batch_size, total_texts) / total_texts * 100
                logger.info(f"Processed {min(i + batch_size, total_texts)}/{total_texts} texts ({progress:.1f}%)")
        
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Generated {strategy.upper()} embeddings: {embeddings_array.shape}")
        
        return embeddings_array
    
    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        strategy: str = 'cls',
        output_prefix: str = 'bert',
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate embeddings for texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            strategy: Embedding strategy ('cls' or 'mean')
            output_prefix: Prefix for embedding column names
            save_path: Optional path to save the result
            
        Returns:
            DataFrame with added embedding columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Generate embeddings
        embeddings = self.embed_batch(
            texts=df[text_column].fillna('').tolist(),
            strategy=strategy
        )
        
        # Create embedding DataFrame
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'{output_prefix}_{strategy}_embed_{i}' for i in range(self.embedding_dim)],
            index=df.index
        )
        
        # Combine with original DataFrame
        result_df = pd.concat([df, embedding_df], axis=1)
        
        if save_path:
            save_dataframe(result_df, save_path)
            logger.info(f"Saved embeddings to {save_path}")
        
        return result_df
    
    def compare_strategies(
        self,
        texts: List[str],
        sample_size: int = 100
    ) -> dict:
        """
        Compare CLS and mean pooling strategies on a sample of texts.
        
        Args:
            texts: List of texts to analyze
            sample_size: Number of texts to sample for comparison
            
        Returns:
            Dictionary with comparison statistics
        """
        if len(texts) > sample_size:
            sample_texts = np.random.choice(texts, sample_size, replace=False).tolist()
        else:
            sample_texts = texts
        
        logger.info(f"Comparing embedding strategies on {len(sample_texts)} texts")
        
        # Generate embeddings with both strategies
        cls_embeddings = self.embed_batch(sample_texts, strategy='cls', show_progress=False)
        mean_embeddings = self.embed_batch(sample_texts, strategy='mean', show_progress=False)
        
        # Calculate statistics
        cls_norms = np.linalg.norm(cls_embeddings, axis=1)
        mean_norms = np.linalg.norm(mean_embeddings, axis=1)
        
        # Calculate cosine similarity between strategies for each text
        similarities = []
        for i in range(len(sample_texts)):
            cls_vec = cls_embeddings[i]
            mean_vec = mean_embeddings[i]
            similarity = np.dot(cls_vec, mean_vec) / (np.linalg.norm(cls_vec) * np.linalg.norm(mean_vec))
            similarities.append(similarity)
        
        comparison_stats = {
            'sample_size': len(sample_texts),
            'cls_avg_norm': np.mean(cls_norms),
            'mean_avg_norm': np.mean(mean_norms),
            'cls_std_norm': np.std(cls_norms),
            'mean_std_norm': np.std(mean_norms),
            'avg_cosine_similarity': np.mean(similarities),
            'std_cosine_similarity': np.std(similarities)
        }
        
        logger.info(f"Strategy comparison: CLS avg norm: {comparison_stats['cls_avg_norm']:.3f}, "
                   f"Mean avg norm: {comparison_stats['mean_avg_norm']:.3f}, "
                   f"Avg similarity: {comparison_stats['avg_cosine_similarity']:.3f}")
        
        return comparison_stats


def main():
    """Example usage of the BertEmbedder class."""
    input_path = Path(config.get('data.processed_data_path')) / "scrubbed_posts.csv"
    
    try:
        # Load data
        df = load_dataframe(input_path)
        logger.info(f"Loaded {len(df)} texts for embedding generation")
        
        # Initialize embedder
        embedder = BertEmbedder()
        
        # Generate CLS embeddings
        cls_output_path = Path(config.get('data.embeddings_path')) / "bert_cls_embeddings.csv"
        cls_df = embedder.embed_dataframe(
            df=df.head(1000),  # Process first 1000 for demo
            text_column='scrubbed_text',
            strategy='cls',
            output_prefix='bert_cls',
            save_path=cls_output_path
        )
        
        # Generate mean embeddings
        mean_output_path = Path(config.get('data.embeddings_path')) / "bert_mean_embeddings.csv"
        mean_df = embedder.embed_dataframe(
            df=df.head(1000),  # Process first 1000 for demo
            text_column='scrubbed_text',
            strategy='mean',
            output_prefix='bert_mean',
            save_path=mean_output_path
        )
        
        # Compare strategies
        comparison = embedder.compare_strategies(df['scrubbed_text'].fillna('').tolist()[:100])
        logger.info(f"Strategy comparison results: {comparison}")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()