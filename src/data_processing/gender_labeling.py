"""
Gender labeling pipeline using GPT-4o-mini for forum post classification.

This module implements the auto-labeling system that tags forum posts as FEMALE, MALE, 
or NEUTRAL using OpenAI's GPT-4o-mini model with chain-of-thought verification.
"""

import openai
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from ..utils.config_loader import config
from ..utils.logging_utils import setup_logger
from ..utils.file_io import save_dataframe, load_dataframe

logger = setup_logger(__name__)


class GenderLabeler:
    """
    Handles gender labeling of forum posts using GPT-4o-mini.
    
    Uses temperature=0 and explicit prompting with examples and chain-of-thought 
    verification to minimize hallucinations. Requires both reasoning and answer 
    fields in the JSON response.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the gender labeler.
        
        Args:
            api_key: OpenAI API key. If None, expects OPENAI_API_KEY environment variable.
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.api_config = config.get_api_config('openai')
        self.model_name = self.api_config.get('model_name', 'gpt-4o-mini')
        self.temperature = self.api_config.get('temperature', 0)
        self.max_tokens = self.api_config.get('max_tokens', 150)
        
        logger.info(f"Initialized GenderLabeler with model: {self.model_name}")
    
    def _create_labeling_prompt(self, text: str) -> str:
        """
        Create the prompt for gender labeling with chain-of-thought reasoning.
        
        Args:
            text: Forum post text to label
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert at analyzing text for gender indicators. Analyze the following forum post and determine if the author is likely FEMALE, MALE, or NEUTRAL (gender cannot be determined).

Look for indicators such as:
- Explicit gender references ("I'm a woman", "as a man", etc.)
- Gendered language patterns
- Cultural/social context clues
- Pronouns referring to self

Text to analyze: "{text}"

Please respond with a JSON object containing:
1. "reasoning": Your step-by-step analysis explaining the gender indicators you found
2. "answer": One of "FEMALE", "MALE", or "NEUTRAL"

Example response format:
{{"reasoning": "The text contains explicit self-reference 'as a woman' indicating female gender.", "answer": "FEMALE"}}

JSON Response:"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Tuple[str, str]:
        """
        Parse the JSON response from the API.
        
        Args:
            response_text: Raw response text from API
            
        Returns:
            Tuple of (reasoning, answer)
            
        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        try:
            # Clean up the response text
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '')
            
            parsed = json.loads(response_text)
            
            reasoning = parsed.get('reasoning', '')
            answer = parsed.get('answer', '').upper()
            
            if answer not in ['FEMALE', 'MALE', 'NEUTRAL']:
                raise ValueError(f"Invalid answer: {answer}")
            
            return reasoning, answer
        
        except (json.JSONError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse response: {response_text[:100]}... Error: {e}")
            return "", "NEUTRAL"
    
    def label_single_text(self, text: str) -> Dict[str, str]:
        """
        Label a single text for gender.
        
        Args:
            text: Text to label
            
        Returns:
            Dictionary with 'gender', 'reasoning', and 'confidence' keys
        """
        if not text or pd.isna(text):
            return {"gender": "NEUTRAL", "reasoning": "Empty text", "confidence": "high"}
        
        try:
            prompt = self._create_labeling_prompt(text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content
            reasoning, answer = self._parse_response(response_text)
            
            return {
                "gender": answer,
                "reasoning": reasoning,
                "confidence": "high" if reasoning else "low"
            }
        
        except Exception as e:
            logger.error(f"Error labeling text: {e}")
            return {"gender": "NEUTRAL", "reasoning": f"Error: {e}", "confidence": "low"}
    
    def label_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        save_progress: bool = True,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Label a batch of texts with progress saving.
        
        Args:
            texts: List of texts to label
            batch_size: Number of texts to process before saving progress
            save_progress: Whether to save progress periodically
            output_path: Path to save progress file
            
        Returns:
            DataFrame with texts and their labels
        """
        if batch_size is None:
            batch_size = config.get('sampling.batch_size', 1000)
        
        if output_path is None:
            output_path = Path(config.get('data.processed_data_path')) / "labeling_progress.csv"
        
        results = []
        total_texts = len(texts)
        
        logger.info(f"Starting batch labeling of {total_texts} texts")
        
        for i, text in enumerate(texts):
            try:
                result = self.label_single_text(text)
                result['text'] = text
                result['index'] = i
                results.append(result)
                
                # Log progress
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / total_texts * 100
                    logger.info(f"Processed {i + 1}/{total_texts} texts ({progress:.1f}%)")
                
                # Save progress periodically
                if save_progress and (i + 1) % batch_size == 0:
                    df_progress = pd.DataFrame(results)
                    save_dataframe(df_progress, output_path)
                    logger.info(f"Saved progress after processing {i + 1} texts")
                
                # Rate limiting to avoid API limits
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                results.append({
                    "text": text,
                    "index": i,
                    "gender": "NEUTRAL",
                    "reasoning": f"Processing error: {e}",
                    "confidence": "low"
                })
        
        df_results = pd.DataFrame(results)
        
        if save_progress:
            save_dataframe(df_results, output_path)
        
        logger.info(f"Completed batch labeling. Results: {df_results['gender'].value_counts().to_dict()}")
        
        return df_results
    
    def validate_labels(self, df: pd.DataFrame, sample_size: int = 500) -> Dict[str, float]:
        """
        Validate labels by re-labeling a random sample and checking agreement.
        
        Args:
            df: DataFrame with existing labels
            sample_size: Number of samples to validate
            
        Returns:
            Dictionary with validation metrics
        """
        if len(df) < sample_size:
            sample_size = len(df)
        
        sample_df = df.sample(n=sample_size, random_state=42)
        
        agreements = 0
        disagreements = 0
        
        logger.info(f"Validating {sample_size} labels")
        
        for _, row in sample_df.iterrows():
            new_result = self.label_single_text(row['text'])
            
            if new_result['gender'] == row['gender']:
                agreements += 1
            else:
                disagreements += 1
                logger.debug(f"Disagreement: '{row['text'][:50]}...' - Original: {row['gender']}, New: {new_result['gender']}")
        
        agreement_rate = agreements / (agreements + disagreements)
        
        validation_results = {
            "agreement_rate": agreement_rate,
            "agreements": agreements,
            "disagreements": disagreements,
            "sample_size": sample_size
        }
        
        logger.info(f"Validation complete. Agreement rate: {agreement_rate:.2%}")
        
        return validation_results


def main():
    """Example usage of the GenderLabeler class."""
    # Load sample data
    input_path = Path(config.get('data.raw_data_path')) / "sample_posts.csv"
    output_path = Path(config.get('data.processed_data_path')) / "labeled_posts.csv"
    
    try:
        df = load_dataframe(input_path)
        logger.info(f"Loaded {len(df)} posts for labeling")
        
        labeler = GenderLabeler()
        
        # Label the texts
        labeled_df = labeler.label_batch(
            texts=df['text'].tolist(),
            output_path=output_path
        )
        
        # Validate a sample
        validation_results = labeler.validate_labels(labeled_df)
        logger.info(f"Validation results: {validation_results}")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()