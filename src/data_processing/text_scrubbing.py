"""
Text scrubbing module for removing explicit gender cues from forum posts.

This module implements the gender cue removal system that substitutes explicit 
gender markers with neutral placeholders like [PRONOUN], [NAME], [FAMILY_MEMBER].
"""

import re
import pandas as pd
from typing import Dict, List, Set, Tuple
import logging
from pathlib import Path

from ..utils.config_loader import config
from ..utils.logging_utils import setup_logger
from ..utils.file_io import save_dataframe, load_dataframe

logger = setup_logger(__name__)


class TextScrubber:
    """
    Handles removal of explicit gender cues from text by replacing them with neutral placeholders.
    
    This class identifies and replaces gendered language elements including:
    - Personal pronouns (he/she, him/her, his/hers)
    - Names (using common name lists)
    - Family relationship terms (brother/sister, husband/wife, etc.)
    - Gendered titles (Mr./Ms., Sir/Madam, etc.)
    """
    
    def __init__(self):
        """Initialize the text scrubber with predefined patterns and replacements."""
        self.placeholders = config.get('placeholders', {})
        
        # Initialize gender-specific patterns
        self._init_pronoun_patterns()
        self._init_name_patterns()
        self._init_family_patterns()
        self._init_title_patterns()
        
        logger.info("Initialized TextScrubber with gender pattern matching")
    
    def _init_pronoun_patterns(self) -> None:
        """Initialize pronoun replacement patterns."""
        self.pronoun_patterns = [
            # Subject pronouns
            (r'\bhe\b', '[PRONOUN]'),
            (r'\bshe\b', '[PRONOUN]'),
            (r'\bHe\b', '[PRONOUN]'),
            (r'\bShe\b', '[PRONOUN]'),
            
            # Object pronouns
            (r'\bhim\b', '[PRONOUN_OBJECT]'),
            (r'\bher\b(?!\s+\w+)', '[PRONOUN_OBJECT]'),  # her as object, not possessive
            (r'\bHim\b', '[PRONOUN_OBJECT]'),
            (r'\bHer\b(?!\s+\w+)', '[PRONOUN_OBJECT]'),
            
            # Possessive pronouns
            (r'\bhis\b', '[PRONOUN_POSSESSIVE]'),
            (r'\bher\s+\w+', '[PRONOUN_POSSESSIVE] \\g<1>'),  # her + noun
            (r'\bhers\b', '[PRONOUN_POSSESSIVE]'),
            (r'\bHis\b', '[PRONOUN_POSSESSIVE]'),
            (r'\bHers\b', '[PRONOUN_POSSESSIVE]'),
            
            # Reflexive pronouns
            (r'\bhimself\b', '[PRONOUN_REFLEXIVE]'),
            (r'\bherself\b', '[PRONOUN_REFLEXIVE]'),
            (r'\bHimself\b', '[PRONOUN_REFLEXIVE]'),
            (r'\bHerself\b', '[PRONOUN_REFLEXIVE]'),
        ]
    
    def _init_name_patterns(self) -> None:
        """Initialize common name patterns for replacement."""
        # Common male names
        male_names = [
            'John', 'Michael', 'David', 'James', 'Robert', 'William', 'Richard', 'Joseph',
            'Thomas', 'Christopher', 'Charles', 'Daniel', 'Matthew', 'Anthony', 'Mark',
            'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian',
            'George', 'Timothy', 'Ronald', 'Jason', 'Edward', 'Jeffrey', 'Ryan', 'Jacob'
        ]
        
        # Common female names
        female_names = [
            'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan',
            'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Helen', 'Sandra',
            'Donna', 'Carol', 'Ruth', 'Sharon', 'Michelle', 'Laura', 'Sarah', 'Kimberly',
            'Deborah', 'Dorothy', 'Lisa', 'Nancy', 'Karen', 'Betty', 'Helen', 'Sandra'
        ]
        
        # Create patterns for names
        all_names = male_names + female_names
        self.name_patterns = []
        
        for name in all_names:
            # Match name as standalone word (case insensitive)
            pattern = f'\\b{re.escape(name)}\\b'
            self.name_patterns.append((pattern, '[NAME]'))
    
    def _init_family_patterns(self) -> None:
        """Initialize family relationship term patterns."""
        self.family_patterns = [
            # Spouse terms
            (r'\bhusband\b', '[FAMILY_MEMBER]'),
            (r'\bwife\b', '[FAMILY_MEMBER]'),
            (r'\bspouse\b', '[FAMILY_MEMBER]'),
            (r'\bHusband\b', '[FAMILY_MEMBER]'),
            (r'\bWife\b', '[FAMILY_MEMBER]'),
            
            # Sibling terms
            (r'\bbrother\b', '[FAMILY_MEMBER]'),
            (r'\bsister\b', '[FAMILY_MEMBER]'),
            (r'\bBrother\b', '[FAMILY_MEMBER]'),
            (r'\bSister\b', '[FAMILY_MEMBER]'),
            
            # Parent terms
            (r'\bfather\b', '[FAMILY_MEMBER]'),
            (r'\bmother\b', '[FAMILY_MEMBER]'),
            (r'\bdad\b', '[FAMILY_MEMBER]'),
            (r'\bmom\b', '[FAMILY_MEMBER]'),
            (r'\bdaddy\b', '[FAMILY_MEMBER]'),
            (r'\bmommy\b', '[FAMILY_MEMBER]'),
            (r'\bFather\b', '[FAMILY_MEMBER]'),
            (r'\bMother\b', '[FAMILY_MEMBER]'),
            
            # Child terms
            (r'\bson\b', '[FAMILY_MEMBER]'),
            (r'\bdaughter\b', '[FAMILY_MEMBER]'),
            (r'\bSon\b', '[FAMILY_MEMBER]'),
            (r'\bDaughter\b', '[FAMILY_MEMBER]'),
            
            # Extended family
            (r'\buncle\b', '[FAMILY_MEMBER]'),
            (r'\baunt\b', '[FAMILY_MEMBER]'),
            (r'\bnephew\b', '[FAMILY_MEMBER]'),
            (r'\bniece\b', '[FAMILY_MEMBER]'),
            (r'\bgrandfather\b', '[FAMILY_MEMBER]'),
            (r'\bgrandmother\b', '[FAMILY_MEMBER]'),
            (r'\bgrandpa\b', '[FAMILY_MEMBER]'),
            (r'\bgrandma\b', '[FAMILY_MEMBER]'),
        ]
    
    def _init_title_patterns(self) -> None:
        """Initialize title and honorific patterns."""
        self.title_patterns = [
            # Formal titles
            (r'\bMr\.\s*', '[TITLE] '),
            (r'\bMrs\.\s*', '[TITLE] '),
            (r'\bMs\.\s*', '[TITLE] '),
            (r'\bMiss\s+', '[TITLE] '),
            (r'\bSir\b', '[TITLE]'),
            (r'\bMadam\b', '[TITLE]'),
            (r'\bMa\'am\b', '[TITLE]'),
            
            # Professional titles with gender implications
            (r'\bwaiter\b', '[TITLE]'),
            (r'\bwaitress\b', '[TITLE]'),
            (r'\bactor\b', '[TITLE]'),
            (r'\bactress\b', '[TITLE]'),
            (r'\bsteward\b', '[TITLE]'),
            (r'\bstewardess\b', '[TITLE]'),
        ]
    
    def scrub_text(self, text: str) -> str:
        """
        Remove gender cues from a single text by replacing with placeholders.
        
        Args:
            text: Input text to scrub
            
        Returns:
            Scrubbed text with gender cues replaced by placeholders
        """
        if not text or pd.isna(text):
            return text
        
        scrubbed = str(text)
        
        # Apply all pattern replacements
        pattern_groups = [
            self.pronoun_patterns,
            self.name_patterns,
            self.family_patterns,
            self.title_patterns
        ]
        
        for pattern_group in pattern_groups:
            for pattern, replacement in pattern_group:
                scrubbed = re.sub(pattern, replacement, scrubbed, flags=re.IGNORECASE)
        
        return scrubbed
    
    def scrub_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        output_column: str = 'scrubbed_text'
    ) -> pd.DataFrame:
        """
        Scrub gender cues from all texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to scrub
            output_column: Name of column to store scrubbed text
            
        Returns:
            DataFrame with additional column containing scrubbed text
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        logger.info(f"Starting text scrubbing for {len(df)} texts")
        
        # Apply scrubbing to all texts
        df[output_column] = df[text_column].apply(self.scrub_text)
        
        # Calculate statistics
        original_lengths = df[text_column].str.len()
        scrubbed_lengths = df[output_column].str.len()
        avg_reduction = (original_lengths - scrubbed_lengths).mean()
        
        logger.info(f"Text scrubbing complete. Average length reduction: {avg_reduction:.1f} characters")
        
        return df
    
    def analyze_scrubbing_impact(
        self,
        df: pd.DataFrame,
        original_column: str = 'text',
        scrubbed_column: str = 'scrubbed_text'
    ) -> Dict[str, float]:
        """
        Analyze the impact of scrubbing on the text corpus.
        
        Args:
            df: DataFrame with original and scrubbed text
            original_column: Name of original text column
            scrubbed_column: Name of scrubbed text column
            
        Returns:
            Dictionary with scrubbing statistics
        """
        original_texts = df[original_column].fillna('')
        scrubbed_texts = df[scrubbed_column].fillna('')
        
        # Calculate various metrics
        original_word_counts = original_texts.str.split().str.len()
        scrubbed_word_counts = scrubbed_texts.str.split().str.len()
        
        # Count placeholder occurrences
        placeholder_counts = {}
        for placeholder in ['[PRONOUN]', '[NAME]', '[FAMILY_MEMBER]', '[TITLE]']:
            count = scrubbed_texts.str.count(re.escape(placeholder)).sum()
            placeholder_counts[placeholder] = count
        
        stats = {
            'total_texts': len(df),
            'avg_original_length': original_texts.str.len().mean(),
            'avg_scrubbed_length': scrubbed_texts.str.len().mean(),
            'avg_word_reduction': (original_word_counts - scrubbed_word_counts).mean(),
            'total_replacements': sum(placeholder_counts.values()),
            'placeholder_counts': placeholder_counts
        }
        
        logger.info(f"Scrubbing analysis: {stats['total_replacements']} total replacements made")
        
        return stats
    
    def validate_scrubbing(
        self,
        df: pd.DataFrame,
        scrubbed_column: str = 'scrubbed_text',
        sample_size: int = 100
    ) -> List[Dict[str, str]]:
        """
        Validate scrubbing by checking for remaining gender cues in a sample.
        
        Args:
            df: DataFrame with scrubbed text
            scrubbed_column: Name of scrubbed text column
            sample_size: Number of texts to validate
            
        Returns:
            List of potential issues found
        """
        if len(df) < sample_size:
            sample_size = len(df)
        
        sample_df = df.sample(n=sample_size, random_state=42)
        issues = []
        
        # Define patterns to check for missed gender cues
        gender_patterns = [
            r'\b(he|she|him|her|his|hers)\b',
            r'\b(husband|wife|brother|sister|son|daughter)\b',
            r'\b(Mr\.|Mrs\.|Ms\.)\b',
            r'\b(man|woman|male|female|guy|girl|boy|lady|gentleman)\b'
        ]
        
        for idx, row in sample_df.iterrows():
            text = str(row[scrubbed_column]).lower()
            
            for pattern in gender_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    issues.append({
                        'index': idx,
                        'text_snippet': text[:100] + '...',
                        'missed_cues': matches,
                        'pattern': pattern
                    })
        
        logger.info(f"Validation complete. Found {len(issues)} potential issues in {sample_size} samples")
        
        return issues


def main():
    """Example usage of the TextScrubber class."""
    input_path = Path(config.get('data.processed_data_path')) / "labeled_posts.csv"
    output_path = Path(config.get('data.processed_data_path')) / "scrubbed_posts.csv"
    
    try:
        # Load labeled data
        df = load_dataframe(input_path)
        logger.info(f"Loaded {len(df)} labeled posts for scrubbing")
        
        # Initialize scrubber
        scrubber = TextScrubber()
        
        # Scrub the texts
        scrubbed_df = scrubber.scrub_dataframe(df, text_column='text')
        
        # Analyze scrubbing impact
        stats = scrubber.analyze_scrubbing_impact(scrubbed_df)
        logger.info(f"Scrubbing statistics: {stats}")
        
        # Validate scrubbing
        issues = scrubber.validate_scrubbing(scrubbed_df)
        if issues:
            logger.warning(f"Found {len(issues)} potential scrubbing issues")
            for issue in issues[:5]:  # Log first 5 issues
                logger.warning(f"Issue: {issue}")
        
        # Save results
        save_dataframe(scrubbed_df, output_path)
        logger.info(f"Saved scrubbed data to {output_path}")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()