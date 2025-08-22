"""
Text scrubbing module for removing explicit gender cues from forum posts.

Uses GPT-4o-mini via the OpenAI API to scrub gender cues while preserving meaning.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import openai
import pandas as pd

from ..utils.config_loader import config
from ..utils.file_io import load_dataframe
from ..utils.file_io import save_dataframe
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class TextScrubber:
    """
    Handles removal of explicit gender cues from text by replacing them with neutral placeholders.

    This class uses an LLM to identify and replace gendered language elements
    (pronouns, names, kinship terms, titles) with placeholders.
    """

    def __init__(self):
        """Initialize the text scrubber. Sets up LLM client and regex patterns."""
        self.placeholders = config.get("placeholders", {})
        # LLM client/config
        self.api_config = config.get("api.openai", {})
        self.model_name = self.api_config.get("model_name", "gpt-4o-mini")
        self.temperature = self.api_config.get("temperature", 0)
        self.max_tokens = self.api_config.get("max_tokens", 300)
        self.client = openai.OpenAI()

        logger.info("Initialized TextScrubber with LLM-based scrubbing")

    # ------------------------
    # LLM-based scrubbing
    # ------------------------

    def _create_scrubbing_prompt(self, text: str) -> str:
        """
        Create prompt instructing the LLM to scrub explicit gender cues about the SUBJECT
        mentioned in the text, replacing with placeholders while preserving meaning.
        """
        prompt = f"""
You are an expert text editor. Your task is to neutralize explicit gender cues about
the people being discussed in a forum post while preserving the original meaning.

Requirements:
- Replace subject-referencing gender cues with these placeholders:
  - he/she -> [PRONOUN]
  - him/her -> [PRONOUN_OBJECT]
  - his/hers OR her <noun> -> [PRONOUN_POSSESSIVE]
  - himself/herself -> [PRONOUN_REFLEXIVE]
  - names -> [NAME]
  - kinship terms (husband/wife, brother/sister, father/mother, son/daughter, etc.) -> [FAMILY_MEMBER]
  - titles and honorifics (Mr., Ms., Sir, Madam, etc.) -> [TITLE]
- Do NOT anonymize non-gendered content.
- Keep punctuation and semantics intact as much as possible.
- If no gender cues exist, return the text unchanged.

Text:
"{text}"

Respond ONLY as JSON like this:
{{
  "scrubbed_text": "...the edited text..."
}}
"""
        return prompt

    def _parse_llm_scrub_response(self, response_text: str) -> Optional[str]:
        """Parse JSON response; tolerate fenced code blocks."""
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            parsed = json.loads(cleaned)
            scrubbed = parsed.get("scrubbed_text")
            if isinstance(scrubbed, str) and len(scrubbed) > 0:
                return scrubbed
            return None
        except Exception:
            return None

    def scrub_text_llm(self, text: str) -> str:
        """Use LLM to scrub a single text. Falls back to original text on failure."""
        if not text or pd.isna(text):
            return text
        try:
            prompt = self._create_scrubbing_prompt(str(text))
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response_text = response.choices[0].message.content
            parsed = self._parse_llm_scrub_response(response_text)
            return parsed if parsed is not None else str(text)
        except Exception as e:
            logger.error(f"LLM scrubbing error: {e}")
            return str(text)

    def scrub_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        output_column: str = "scrubbed_text",
        batch_sleep_seconds: float = 0.05,
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

        results: list[str] = []
        for idx, original in enumerate(df[text_column].tolist()):
            results.append(self.scrub_text_llm(original))
            if (idx + 1) % 10 == 0:
                logger.debug(f"LLM scrubbed {idx + 1}/{len(df)} texts")
            time.sleep(batch_sleep_seconds)
        df[output_column] = results

        # Calculate statistics
        original_lengths = df[text_column].str.len()
        scrubbed_lengths = df[output_column].str.len()
        avg_reduction = (original_lengths - scrubbed_lengths).mean()

        logger.info(
            f"Text scrubbing complete. Average length reduction: {avg_reduction:.1f} characters"
        )

        return df

    def analyze_scrubbing_impact(
        self,
        df: pd.DataFrame,
        original_column: str = "text",
        scrubbed_column: str = "scrubbed_text",
    ) -> dict[str, float]:
        """
        Analyze the impact of scrubbing on the text corpus.

        Args:
            df: DataFrame with original and scrubbed text
            original_column: Name of original text column
            scrubbed_column: Name of scrubbed text column

        Returns:
            Dictionary with scrubbing statistics
        """
        original_texts = df[original_column].fillna("")
        scrubbed_texts = df[scrubbed_column].fillna("")

        # Calculate various metrics
        original_word_counts = original_texts.str.split().str.len()
        scrubbed_word_counts = scrubbed_texts.str.split().str.len()

        # Count placeholder occurrences
        placeholder_counts = {}
        for placeholder in ["[PRONOUN]", "[NAME]", "[FAMILY_MEMBER]", "[TITLE]"]:
            count = scrubbed_texts.str.count(re.escape(placeholder)).sum()
            placeholder_counts[placeholder] = count

        stats = {
            "total_texts": len(df),
            "avg_original_length": original_texts.str.len().mean(),
            "avg_scrubbed_length": scrubbed_texts.str.len().mean(),
            "avg_word_reduction": (original_word_counts - scrubbed_word_counts).mean(),
            "total_replacements": sum(placeholder_counts.values()),
            "placeholder_counts": placeholder_counts,
        }

        logger.info(
            f"Scrubbing analysis: {stats['total_replacements']} total replacements made"
        )

        return stats

    def validate_scrubbing(
        self,
        df: pd.DataFrame,
        scrubbed_column: str = "scrubbed_text",
        sample_size: int = 100,
    ) -> list[dict[str, str]]:
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
            r"\b(he|she|him|her|his|hers)\b",
            r"\b(husband|wife|brother|sister|son|daughter)\b",
            r"\b(Mr\.|Mrs\.|Ms\.)\b",
            r"\b(man|woman|male|female|guy|girl|boy|lady|gentleman)\b",
        ]

        for idx, row in sample_df.iterrows():
            text = str(row[scrubbed_column]).lower()

            for pattern in gender_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    issues.append(
                        {
                            "index": idx,
                            "text_snippet": text[:100] + "...",
                            "missed_cues": matches,
                            "pattern": pattern,
                        }
                    )

        logger.info(
            f"Validation complete. Found {len(issues)} potential issues in {sample_size} samples"
        )

        return issues


def main():
    """Example usage of the TextScrubber class."""
    input_path = Path(config.get("data.processed_data_path")) / "labeled_posts.csv"
    output_path = Path(config.get("data.processed_data_path")) / "scrubbed_posts.csv"

    try:
        # Load labeled data
        df = load_dataframe(input_path)
        logger.info(f"Loaded {len(df)} labeled posts for scrubbing")

        # Initialize scrubber
        scrubber = TextScrubber()

        # Scrub the texts
        scrubbed_df = scrubber.scrub_dataframe(df, text_column="text")

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
