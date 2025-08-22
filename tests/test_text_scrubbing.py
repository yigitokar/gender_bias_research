"""
Comprehensive unit tests for text scrubbing module.

Tests cover pronoun replacement, name masking, family term substitution,
LLM-based scrubbing, error handling, and edge cases. Demonstrates
production-grade testing with mocking and exception handling.
"""

import json
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_processing.text_scrubbing import TextScrubber


class TestTextScrubber:
    """Test suite for TextScrubber class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client for LLM scrubbing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Default scrubbed response
        mock_message.content = json.dumps(
            {
                "scrubbed_text": "[PRONOUN] told [PRONOUN_POSSESSIVE] [FAMILY_MEMBER] about the news."
            }
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    @pytest.fixture
    def scrubber(self, mock_openai_client):
        """Create a TextScrubber instance with mocked OpenAI client."""
        with patch(
            "src.data_processing.text_scrubbing.openai.OpenAI",
            return_value=mock_openai_client,
        ):
            return TextScrubber()

    def test_initialization(self, scrubber):
        """Test proper initialization of TextScrubber."""
        assert scrubber.model_name == "gpt-4o-mini"
        assert scrubber.temperature == 0
        assert hasattr(scrubber, "client")

    def test_scrub_single_text_pronouns(self, scrubber):
        """Test scrubbing of pronouns in text."""
        text = "He told her that his work was important to him."
        result = scrubber.scrub_single_text(text)

        assert "[PRONOUN]" in result
        assert "he" not in result.lower()
        assert "her" not in result.lower()
        assert "his" not in result.lower()

    def test_scrub_single_text_names(self, scrubber, mock_openai_client):
        """Test scrubbing of names in text."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {"scrubbed_text": "[NAME] and [NAME] went to the meeting."}
        )

        text = "John and Mary went to the meeting."
        result = scrubber.scrub_single_text(text)

        assert "[NAME]" in result
        assert "John" not in result
        assert "Mary" not in result

    def test_scrub_single_text_family_terms(self, scrubber, mock_openai_client):
        """Test scrubbing of family terms."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {"scrubbed_text": "My [FAMILY_MEMBER] and [FAMILY_MEMBER] are visiting."}
        )

        text = "My mother and father are visiting."
        result = scrubber.scrub_single_text(text)

        assert "[FAMILY_MEMBER]" in result
        assert "mother" not in result.lower()
        assert "father" not in result.lower()

    def test_scrub_empty_text(self, scrubber):
        """Test handling of empty text."""
        result = scrubber.scrub_single_text("")
        assert result == ""

    def test_scrub_none_text(self, scrubber):
        """Test handling of None text."""
        result = scrubber.scrub_single_text(None)
        assert result == ""

    def test_scrub_nan_text(self, scrubber):
        """Test handling of NaN text."""
        result = scrubber.scrub_single_text(pd.NA)
        assert result == ""

    def test_api_error_handling(self, scrubber, mock_openai_client):
        """Test handling of API errors during scrubbing."""
        # Simulate API error
        mock_openai_client.chat.completions.create.side_effect = Exception(
            "API rate limit"
        )

        text = "She is working on her project."
        result = scrubber.scrub_single_text(text)

        # Should return original text on error
        assert result == text

    def test_malformed_json_response(self, scrubber, mock_openai_client):
        """Test handling of malformed JSON responses."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "Invalid JSON"

        text = "Test text for scrubbing."
        result = scrubber.scrub_single_text(text)

        # Should return original text on parse error
        assert result == text

    def test_json_with_code_blocks(self, scrubber, mock_openai_client):
        """Test parsing JSON wrapped in code blocks."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = """```json
        {
            "scrubbed_text": "[PRONOUN] is working."
        }
        ```"""

        text = "She is working."
        result = scrubber.scrub_single_text(text)

        assert result == "[PRONOUN] is working."

    def test_missing_scrubbed_text_field(self, scrubber, mock_openai_client):
        """Test handling of missing 'scrubbed_text' field."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps({"other_field": "some value"})

        text = "Original text."
        result = scrubber.scrub_single_text(text)

        # Should return original text if field is missing
        assert result == text

    def test_scrub_dataframe(self, scrubber, tmp_path):
        """Test scrubbing an entire dataframe."""
        df = pd.DataFrame(
            {
                "text": ["She is happy.", "He is working.", "They are studying."],
                "label": ["FEMALE", "MALE", "NEUTRAL"],
            }
        )

        with patch("src.data_processing.text_scrubbing.save_dataframe") as mock_save:
            result_df = scrubber.scrub_dataframe(
                df,
                text_column="text",
                output_column="scrubbed",
                save_progress=True,
                output_path=tmp_path / "progress.csv",
            )

        assert len(result_df) == 3
        assert "scrubbed" in result_df.columns
        assert all(col in result_df.columns for col in df.columns)

    def test_batch_scrubbing_with_errors(self, scrubber, mock_openai_client):
        """Test batch scrubbing handles individual errors gracefully."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Make second call fail
        side_effects = [
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content=json.dumps({"scrubbed_text": "Scrubbed 1"})
                        )
                    )
                ]
            ),
            Exception("API Error"),
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content=json.dumps({"scrubbed_text": "Scrubbed 3"})
                        )
                    )
                ]
            ),
        ]
        mock_openai_client.chat.completions.create.side_effect = side_effects

        results = scrubber.scrub_batch(texts)

        assert len(results) == 3
        assert results[0] == "Scrubbed 1"
        assert results[1] == "Text 2"  # Original text on error
        assert results[2] == "Scrubbed 3"

    def test_analyze_scrubbing(self, scrubber):
        """Test scrubbing analysis functionality."""
        original_texts = ["She told her sister.", "He works.", "They study."]
        scrubbed_texts = [
            "[PRONOUN] told [PRONOUN_POSSESSIVE] [FAMILY_MEMBER].",
            "[PRONOUN] works.",
            "They study.",
        ]

        analysis = scrubber.analyze_scrubbing(original_texts, scrubbed_texts)

        assert "total_texts" in analysis
        assert "texts_modified" in analysis
        assert "modification_rate" in analysis
        assert "avg_words_replaced" in analysis
        assert "max_words_replaced" in analysis
        assert "examples_heavily_modified" in analysis

        assert analysis["total_texts"] == 3
        assert analysis["texts_modified"] == 2

    def test_validate_scrubbing(self, scrubber):
        """Test validation of scrubbing for missed gender cues."""
        scrubbed_texts = [
            "[PRONOUN] is working.",
            "She is studying.",  # Missed pronoun
            "[NAME] went home.",
        ]

        validation_results = scrubber.validate_scrubbing(scrubbed_texts)

        assert "texts_with_issues" in validation_results
        assert "total_issues" in validation_results
        assert "issue_rate" in validation_results
        assert "examples" in validation_results

        assert validation_results["total_issues"] >= 1
        assert any("She" in ex["text"] for ex in validation_results["examples"])

    def test_validate_scrubbing_no_issues(self, scrubber):
        """Test validation when no gender cues are missed."""
        scrubbed_texts = [
            "[PRONOUN] is working.",
            "[NAME] is studying.",
            "The meeting was productive.",
        ]

        validation_results = scrubber.validate_scrubbing(scrubbed_texts)

        assert validation_results["total_issues"] == 0
        assert validation_results["issue_rate"] == 0.0

    def test_prompt_creation(self, scrubber):
        """Test that scrubbing prompt is created correctly."""
        text = "She told her mother about the job."
        prompt = scrubber._create_scrubbing_prompt(text)

        assert text in prompt
        assert "[PRONOUN]" in prompt
        assert "[NAME]" in prompt
        assert "[FAMILY_MEMBER]" in prompt
        assert "JSON" in prompt
        assert "scrubbed_text" in prompt

    def test_parse_llm_response_edge_cases(self, scrubber):
        """Test LLM response parsing with edge cases."""
        # Test with extra whitespace
        result = scrubber._parse_llm_scrub_response('  {"scrubbed_text": "Test"}  ')
        assert result == "Test"

        # Test empty response
        result = scrubber._parse_llm_scrub_response("")
        assert result is None

        # Test non-string scrubbed_text
        result = scrubber._parse_llm_scrub_response('{"scrubbed_text": 123}')
        assert result is None

    def test_case_preservation(self, scrubber, mock_openai_client):
        """Test that case is preserved in scrubbing."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {"scrubbed_text": "[PRONOUN] told [PRONOUN_POSSESSIVE] friend."}
        )

        text = "HE told HIS friend."
        result = scrubber.scrub_single_text(text)

        # Check that placeholders maintain reasonable formatting
        assert "[PRONOUN]" in result

    def test_complex_text_scrubbing(self, scrubber, mock_openai_client):
        """Test scrubbing of complex text with multiple gender indicators."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "scrubbed_text": "[NAME] told [PRONOUN_POSSESSIVE] [FAMILY_MEMBER] that [PRONOUN] would visit [PRONOUN_POSSESSIVE] [FAMILY_MEMBER] tomorrow."
            }
        )

        text = "Sarah told her mother that she would visit her grandmother tomorrow."
        result = scrubber.scrub_single_text(text)

        assert "Sarah" not in result
        assert "mother" not in result
        assert "grandmother" not in result
        assert "she" not in result.lower()
        assert "her" not in result.lower()

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_rate_limiting(self, mock_sleep, scrubber):
        """Test that rate limiting is applied during batch processing."""
        texts = ["Text 1", "Text 2"]

        scrubber.scrub_batch(texts)

        # Check that sleep was called for rate limiting
        assert mock_sleep.call_count >= len(texts)

    def test_dataframe_with_missing_values(self, scrubber):
        """Test scrubbing dataframe with missing values."""
        df = pd.DataFrame(
            {"text": ["Valid text", None, pd.NA, "Another text"], "id": [1, 2, 3, 4]}
        )

        result_df = scrubber.scrub_dataframe(
            df, text_column="text", save_progress=False
        )

        assert len(result_df) == 4
        assert result_df.iloc[1]["scrubbed_text"] == ""
        assert result_df.iloc[2]["scrubbed_text"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
