"""
Unit tests for the text scrubbing module.

This module demonstrates proper testing practices for the gender cue removal functionality.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module we're testing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.text_scrubbing import TextScrubber


class TestTextScrubber(unittest.TestCase):
    """Test cases for the TextScrubber class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.scrubber = TextScrubber()
        
        # Test texts with various gender cues
        self.test_texts = {
            'pronouns': "He went to the store and she followed him.",
            'names': "John called Mary to discuss the project with Michael.",
            'family': "My husband and his sister visited my mother yesterday.",
            'titles': "Mr. Smith met with Ms. Johnson and Mrs. Davis.",
            'mixed': "She told her brother that his wife, Sarah, works with Mr. Brown.",
            'neutral': "The person went to the meeting to discuss the proposal.",
            'empty': "",
            'none_value': None
        }
        
        # Expected results after scrubbing
        self.expected_results = {
            'pronouns': "[PRONOUN] went to the store and [PRONOUN] followed [PRONOUN_OBJECT].",
            'names': "[NAME] called [NAME] to discuss the project with [NAME].",
            'family': "My [FAMILY_MEMBER] and [PRONOUN_POSSESSIVE] [FAMILY_MEMBER] visited my [FAMILY_MEMBER] yesterday.",
            'titles': "[TITLE] [NAME] met with [TITLE] [NAME] and [TITLE] [NAME].",
            'neutral': "The person went to the meeting to discuss the proposal.",
            'empty': "",
            'none_value': None
        }
    
    def test_scrub_pronouns(self):
        """Test that pronouns are correctly replaced with placeholders."""
        test_text = "He said she would help him with his work."
        result = self.scrubber.scrub_text(test_text)
        
        # Check that pronouns are replaced
        self.assertNotIn('He', result)
        self.assertNotIn('she', result)
        self.assertNotIn('him', result)
        self.assertNotIn('his', result)
        
        # Check that placeholders are present
        self.assertIn('[PRONOUN]', result)
        self.assertIn('[PRONOUN_OBJECT]', result)
        self.assertIn('[PRONOUN_POSSESSIVE]', result)
    
    def test_scrub_names(self):
        """Test that common names are replaced with placeholders."""
        test_text = "John and Mary went to see Michael."
        result = self.scrubber.scrub_text(test_text)
        
        # Check that names are replaced
        self.assertNotIn('John', result)
        self.assertNotIn('Mary', result)
        self.assertNotIn('Michael', result)
        
        # Check that placeholder is present
        self.assertIn('[NAME]', result)
    
    def test_scrub_family_terms(self):
        """Test that family relationship terms are replaced."""
        test_text = "My husband and wife went with their son and daughter."
        result = self.scrubber.scrub_text(test_text)
        
        # Check that family terms are replaced
        self.assertNotIn('husband', result)
        self.assertNotIn('wife', result)
        self.assertNotIn('son', result)
        self.assertNotIn('daughter', result)
        
        # Check that placeholder is present
        self.assertIn('[FAMILY_MEMBER]', result)
    
    def test_scrub_titles(self):
        """Test that gendered titles are replaced."""
        test_text = "Mr. Smith and Mrs. Johnson met with Ms. Davis."
        result = self.scrubber.scrub_text(test_text)
        
        # Check that titles are replaced
        self.assertNotIn('Mr.', result)
        self.assertNotIn('Mrs.', result)
        self.assertNotIn('Ms.', result)
        
        # Check that placeholder is present
        self.assertIn('[TITLE]', result)
    
    def test_case_insensitive_replacement(self):
        """Test that replacement works regardless of case."""
        test_cases = [
            ("He went home", "[PRONOUN] went home"),
            ("HE went home", "[PRONOUN] went home"),
            ("he went home", "[PRONOUN] went home"),
            ("JOHN called", "[NAME] called"),
            ("john called", "[NAME] called")
        ]
        
        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                result = self.scrubber.scrub_text(input_text)
                self.assertEqual(result, expected_output)
    
    def test_word_boundaries(self):
        """Test that replacement respects word boundaries."""
        # Words containing gender terms but not actual gender terms
        test_text = "The theory and theorem were discussed in therapy."
        result = self.scrubber.scrub_text(test_text)
        
        # These should NOT be replaced because they're not whole words
        self.assertIn('theory', result)
        self.assertIn('theorem', result)
        self.assertIn('therapy', result)
    
    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        # Test empty string
        result_empty = self.scrubber.scrub_text("")
        self.assertEqual(result_empty, "")
        
        # Test None
        result_none = self.scrubber.scrub_text(None)
        self.assertIsNone(result_none)
        
        # Test NaN
        result_nan = self.scrubber.scrub_text(pd.NA)
        self.assertTrue(pd.isna(result_nan))
    
    def test_scrub_dataframe(self):
        """Test scrubbing functionality on a DataFrame."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'text': [
                "He said hello to Mary.",
                "She met with Mr. Johnson.",
                "The person attended the meeting."
            ],
            'other_column': ['A', 'B', 'C']
        })
        
        # Apply scrubbing
        result_df = self.scrubber.scrub_dataframe(test_df, 'text', 'scrubbed_text')
        
        # Check that original DataFrame is preserved
        self.assertTrue('text' in result_df.columns)
        self.assertTrue('other_column' in result_df.columns)
        
        # Check that scrubbed column is added
        self.assertTrue('scrubbed_text' in result_df.columns)
        
        # Check that scrubbing was applied
        scrubbed_texts = result_df['scrubbed_text'].tolist()
        self.assertNotIn('He', ' '.join(scrubbed_texts))
        self.assertNotIn('Mary', ' '.join(scrubbed_texts))
        self.assertNotIn('Mr.', ' '.join(scrubbed_texts))
    
    def test_scrub_dataframe_missing_column(self):
        """Test that scrub_dataframe raises error for missing column."""
        test_df = pd.DataFrame({'other_column': ['A', 'B', 'C']})
        
        with self.assertRaises(ValueError):
            self.scrubber.scrub_dataframe(test_df, 'missing_column')
    
    def test_analyze_scrubbing_impact(self):
        """Test the scrubbing impact analysis functionality."""
        # Create test DataFrame with before/after scrubbing
        test_df = pd.DataFrame({
            'text': [
                "He told Mary about his work.",
                "She visited her mother yesterday.",
                "The meeting was productive."
            ],
            'scrubbed_text': [
                "[PRONOUN] told [NAME] about [PRONOUN_POSSESSIVE] work.",
                "[PRONOUN] visited [PRONOUN_POSSESSIVE] [FAMILY_MEMBER] yesterday.",
                "The meeting was productive."
            ]
        })
        
        # Analyze impact
        stats = self.scrubber.analyze_scrubbing_impact(test_df, 'text', 'scrubbed_text')
        
        # Check that statistics are calculated
        self.assertIn('total_texts', stats)
        self.assertIn('avg_original_length', stats)
        self.assertIn('avg_scrubbed_length', stats)
        self.assertIn('total_replacements', stats)
        self.assertIn('placeholder_counts', stats)
        
        # Check specific values
        self.assertEqual(stats['total_texts'], 3)
        self.assertGreater(stats['total_replacements'], 0)
    
    def test_validate_scrubbing(self):
        """Test the scrubbing validation functionality."""
        # Create test DataFrame with some missed gender cues
        test_df = pd.DataFrame({
            'scrubbed_text': [
                "[PRONOUN] told [NAME] about the work.",  # Good scrubbing
                "The guy met with her yesterday.",        # Missed 'guy' and 'her'
                "The meeting was productive."             # No gender cues
            ]
        })
        
        # Validate scrubbing
        issues = self.scrubber.validate_scrubbing(test_df, 'scrubbed_text', sample_size=3)
        
        # Should find at least one issue (the second text)
        self.assertGreater(len(issues), 0)
        
        # Check issue structure
        if issues:
            issue = issues[0]
            self.assertIn('index', issue)
            self.assertIn('text_snippet', issue)
            self.assertIn('missed_cues', issue)
    
    @patch('src.data_processing.text_scrubbing.logger')
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made."""
        # Create test DataFrame
        test_df = pd.DataFrame({'text': ["He said hello."]})
        
        # Call scrub_dataframe
        self.scrubber.scrub_dataframe(test_df, 'text')
        
        # Check that info logging was called
        mock_logger.info.assert_called()
    
    def test_complex_text_scrubbing(self):
        """Test scrubbing on complex text with multiple gender cues."""
        complex_text = """
        Yesterday, John called his wife Mary to tell her that he would be late. 
        She understood and told him that her mother was visiting. 
        Mr. Johnson, their neighbor, had offered to help with the children.
        His daughter Sarah and her husband were also coming to dinner.
        """
        
        result = self.scrubber.scrub_text(complex_text)
        
        # Check that all gender cues are removed
        gender_cues = ['John', 'his', 'wife', 'Mary', 'her', 'he', 'She', 'him', 
                      'Mr.', 'Johnson', 'His', 'daughter', 'Sarah', 'husband']
        
        for cue in gender_cues:
            self.assertNotIn(cue, result, f"Gender cue '{cue}' was not removed")
        
        # Check that placeholders are present
        placeholders = ['[NAME]', '[PRONOUN]', '[PRONOUN_POSSESSIVE]', 
                       '[PRONOUN_OBJECT]', '[FAMILY_MEMBER]', '[TITLE]']
        
        for placeholder in placeholders:
            self.assertIn(placeholder, result, f"Placeholder '{placeholder}' is missing")


class TestTextScrubberIntegration(unittest.TestCase):
    """Integration tests for TextScrubber with mocked dependencies."""
    
    @patch('src.data_processing.text_scrubbing.config')
    def test_initialization_with_config(self, mock_config):
        """Test TextScrubber initialization with mocked config."""
        # Mock config response
        mock_config.get.return_value = {
            'pronouns': ['[PRONOUN]'],
            'names': '[NAME]',
            'family_members': '[FAMILY_MEMBER]',
            'titles': '[TITLE]'
        }
        
        # Initialize scrubber
        scrubber = TextScrubber()
        
        # Verify config was called
        mock_config.get.assert_called_with('placeholders', {})
    
    @patch('src.data_processing.text_scrubbing.save_dataframe')
    @patch('src.data_processing.text_scrubbing.load_dataframe')
    def test_main_function_integration(self, mock_load, mock_save):
        """Test the main function with mocked file I/O."""
        # Mock loaded DataFrame
        mock_df = pd.DataFrame({
            'text': ["He said hello to Mary.", "She went home."],
            'gender': ['MALE', 'FEMALE']
        })
        mock_load.return_value = mock_df
        
        # Import and run main (would need to be in the actual module)
        # This is a placeholder for how you might test the main function
        pass


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextScrubber))
    suite.addTests(loader.loadTestsFromTestCase(TestTextScrubberIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    # Run the tests
    test_result = run_tests()
    
    # Exit with appropriate code
    exit_code = 0 if test_result.wasSuccessful() else 1
    sys.exit(exit_code)