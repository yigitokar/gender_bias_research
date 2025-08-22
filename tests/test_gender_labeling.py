"""
Comprehensive unit tests for gender labeling module.

Tests cover successful labeling, error handling, API failures, JSON parsing edge cases,
and batch processing with progress saving. Demonstrates production-grade testing practices.
"""

import json
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_processing.gender_labeling import GenderLabeler


class TestGenderLabeler:
    """Test suite for GenderLabeler class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Set up the chain of mock objects
        mock_message.content = json.dumps(
            {
                "reasoning": "The text mentions 'she' and 'her', indicating female subject.",
                "answer": "FEMALE",
            }
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    @pytest.fixture
    def labeler(self, mock_openai_client):
        """Create a GenderLabeler instance with mocked OpenAI client."""
        with patch(
            "src.data_processing.gender_labeling.openai.OpenAI",
            return_value=mock_openai_client,
        ):
            return GenderLabeler(api_key="test_key")

    def test_initialization(self, labeler):
        """Test proper initialization of GenderLabeler."""
        assert labeler.model_name == "gpt-4o-mini"
        assert labeler.temperature == 0
        assert labeler.max_tokens == 150

    def test_label_single_text_female(self, labeler):
        """Test successful labeling of female-oriented text."""
        text = "She told her sister about the exciting news."
        result = labeler.label_single_text(text)

        assert result["gender"] == "FEMALE"
        assert "reasoning" in result
        assert result["confidence"] == "high"

    def test_label_single_text_male(self, labeler, mock_openai_client):
        """Test successful labeling of male-oriented text."""
        # Update mock response for male
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "reasoning": "The text mentions 'he' and 'his', indicating male subject.",
                "answer": "MALE",
            }
        )

        text = "He told his brother about the game."
        result = labeler.label_single_text(text)

        assert result["gender"] == "MALE"
        assert "reasoning" in result
        assert result["confidence"] == "high"

    def test_label_single_text_neutral(self, labeler, mock_openai_client):
        """Test labeling of neutral/ambiguous text."""
        # Update mock response for neutral
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "reasoning": "No clear gender indicators found in the text.",
                "answer": "NEUTRAL",
            }
        )

        text = "The meeting was productive and everyone contributed."
        result = labeler.label_single_text(text)

        assert result["gender"] == "NEUTRAL"
        assert result["confidence"] == "high"

    def test_label_empty_text(self, labeler):
        """Test handling of empty text input."""
        result = labeler.label_single_text("")

        assert result["gender"] == "NEUTRAL"
        assert result["reasoning"] == "Empty text"
        assert result["confidence"] == "high"

    def test_label_none_text(self, labeler):
        """Test handling of None text input."""
        result = labeler.label_single_text(None)

        assert result["gender"] == "NEUTRAL"
        assert result["reasoning"] == "Empty text"
        assert result["confidence"] == "high"

    def test_label_nan_text(self, labeler):
        """Test handling of NaN text input."""
        result = labeler.label_single_text(pd.NA)

        assert result["gender"] == "NEUTRAL"
        assert result["reasoning"] == "Empty text"
        assert result["confidence"] == "high"

    def test_api_error_handling(self, labeler, mock_openai_client):
        """Test handling of API errors."""
        # Simulate API error
        mock_openai_client.chat.completions.create.side_effect = Exception(
            "API rate limit exceeded"
        )

        text = "This is a test text."
        result = labeler.label_single_text(text)

        assert result["gender"] == "NEUTRAL"
        assert "Error: API rate limit exceeded" in result["reasoning"]
        assert result["confidence"] == "low"

    def test_malformed_json_response(self, labeler, mock_openai_client):
        """Test handling of malformed JSON responses."""
        # Return invalid JSON
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "Not valid JSON"

        text = "Test text for JSON parsing."
        result = labeler.label_single_text(text)

        assert result["gender"] == "NEUTRAL"
        assert result["confidence"] == "low"

    def test_json_with_code_blocks(self, labeler, mock_openai_client):
        """Test parsing JSON wrapped in code blocks."""
        # JSON wrapped in markdown code blocks
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = """```json
        {
            "reasoning": "Subject is identified as female.",
            "answer": "FEMALE"
        }
        ```"""

        text = "She is a talented engineer."
        result = labeler.label_single_text(text)

        assert result["gender"] == "FEMALE"
        assert result["confidence"] == "high"

    def test_invalid_gender_answer(self, labeler, mock_openai_client):
        """Test handling of invalid gender values in response."""
        # Return invalid gender value
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "reasoning": "Some reasoning",
                "answer": "UNKNOWN",  # Invalid value
            }
        )

        text = "Test text."
        result = labeler.label_single_text(text)

        assert result["gender"] == "NEUTRAL"
        assert result["confidence"] == "low"

    def test_missing_fields_in_response(self, labeler, mock_openai_client):
        """Test handling of missing fields in JSON response."""
        # Missing 'answer' field
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps({"reasoning": "Some reasoning"})

        text = "Test text."
        result = labeler.label_single_text(text)

        assert result["gender"] == "NEUTRAL"
        assert result["confidence"] == "low"

    def test_batch_labeling(self, labeler, tmp_path):
        """Test batch labeling with progress saving."""
        texts = ["She is happy.", "He is working.", "The weather is nice."]

        output_path = tmp_path / "test_progress.csv"

        with patch("src.data_processing.gender_labeling.save_dataframe") as mock_save:
            result_df = labeler.label_batch(
                texts=texts,
                batch_size=2,
                save_progress=True,
                output_path=str(output_path),
            )

        assert len(result_df) == 3
        assert "gender" in result_df.columns
        assert "reasoning" in result_df.columns
        assert "confidence" in result_df.columns
        assert "text" in result_df.columns
        assert "index" in result_df.columns

        # Check that save was called (once for batch, once for final)
        assert mock_save.call_count >= 1

    def test_batch_labeling_with_errors(self, labeler, mock_openai_client, tmp_path):
        """Test batch labeling handles individual errors gracefully."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Make second call fail
        side_effects = [
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content=json.dumps({"reasoning": "R1", "answer": "FEMALE"})
                        )
                    )
                ]
            ),
            Exception("API Error"),
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content=json.dumps({"reasoning": "R3", "answer": "MALE"})
                        )
                    )
                ]
            ),
        ]
        mock_openai_client.chat.completions.create.side_effect = side_effects

        output_path = tmp_path / "test_progress.csv"

        with patch("src.data_processing.gender_labeling.save_dataframe"):
            result_df = labeler.label_batch(
                texts=texts,
                batch_size=10,
                save_progress=False,
                output_path=str(output_path),
            )

        assert len(result_df) == 3
        assert result_df.iloc[0]["gender"] == "FEMALE"
        assert result_df.iloc[1]["gender"] == "NEUTRAL"  # Error default
        assert result_df.iloc[2]["gender"] == "MALE"

    def test_validate_labels(self, labeler):
        """Test label validation functionality."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "text": ["She is happy.", "He is working.", "They are studying."],
                "gender": ["FEMALE", "MALE", "NEUTRAL"],
            }
        )

        # Mock re-labeling to match original labels
        validation_results = labeler.validate_labels(df, sample_size=2)

        assert "agreement_rate" in validation_results
        assert "agreements" in validation_results
        assert "disagreements" in validation_results
        assert "sample_size" in validation_results
        assert validation_results["sample_size"] <= 2

    def test_validate_labels_small_dataset(self, labeler):
        """Test validation with dataset smaller than sample size."""
        # Create small test dataframe
        df = pd.DataFrame({"text": ["Text 1"], "gender": ["FEMALE"]})

        validation_results = labeler.validate_labels(df, sample_size=100)

        assert validation_results["sample_size"] == 1

    def test_prompt_creation(self, labeler):
        """Test that prompt is created correctly."""
        text = "This is a test text."
        prompt = labeler._create_labeling_prompt(text)

        assert text in prompt
        assert "FEMALE" in prompt
        assert "MALE" in prompt
        assert "NEUTRAL" in prompt
        assert "JSON" in prompt
        assert "reasoning" in prompt
        assert "answer" in prompt

    def test_parse_response_edge_cases(self, labeler):
        """Test response parsing with various edge cases."""
        # Test uppercase conversion
        reasoning, answer = labeler._parse_response(
            '{"reasoning": "test", "answer": "female"}'
        )
        assert answer == "FEMALE"

        # Test with extra whitespace
        reasoning, answer = labeler._parse_response(
            '  {"reasoning": "test", "answer": "MALE"}  '
        )
        assert answer == "MALE"

        # Test empty response
        reasoning, answer = labeler._parse_response("")
        assert answer == "NEUTRAL"
        assert reasoning == ""

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_rate_limiting(self, mock_sleep, labeler):
        """Test that rate limiting is applied during batch processing."""
        texts = ["Text 1", "Text 2"]

        with patch("src.data_processing.gender_labeling.save_dataframe"):
            labeler.label_batch(texts=texts, save_progress=False)

        # Check that sleep was called for rate limiting
        assert mock_sleep.call_count >= len(texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
