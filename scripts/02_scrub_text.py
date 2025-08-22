#!/usr/bin/env python3
"""
Stage 2: Text Scrubbing Pipeline

This script removes explicit gender cues from labeled forum posts by substituting
neutral placeholders like [PRONOUN], [NAME], [FAMILY_MEMBER].

Usage:
    python scripts/02_scrub_text.py --input data/processed/labeled_posts.csv --output data/processed/scrubbed_posts.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.text_scrubbing import TextScrubber
from src.utils.file_io import ensure_directory
from src.utils.file_io import load_dataframe
from src.utils.file_io import save_dataframe
from src.utils.file_io import save_json
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove explicit gender cues from labeled forum posts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with labeled posts",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save scrubbed output CSV file",
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="raw_cat",
        help="Name of column containing text to scrub",
    )

    parser.add_argument(
        "--output-column",
        type=str,
        default="scrubbed_text",
        help="Name of column to store scrubbed text",
    )

    # LLM-only; method flag removed

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform validation to check for missed gender cues",
    )

    parser.add_argument(
        "--validation-size", type=int, default=100, help="Number of samples to validate"
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform detailed analysis of scrubbing impact",
    )

    parser.add_argument(
        "--filter-gender",
        type=str,
        choices=["MALE", "FEMALE", "NEUTRAL"],
        help="Filter to only process posts with specific gender label",
    )

    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments and files."""
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    output_path = Path(args.output)
    ensure_directory(output_path.parent)

    return input_path, output_path


def filter_data(df, text_column, filter_gender=None):
    """
    Filter and prepare data for scrubbing.

    Args:
        df: Input DataFrame
        text_column: Name of text column
        filter_gender: Optional gender filter

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Original dataset shape: {df.shape}")

    # Remove rows with missing text
    df_filtered = df.dropna(subset=[text_column])
    logger.info(f"After removing missing text: {df_filtered.shape}")

    # Filter by gender if specified
    if filter_gender and "gender" in df.columns:
        df_filtered = df_filtered[df_filtered["gender"] == filter_gender]
        logger.info(f"After filtering for {filter_gender} posts: {df_filtered.shape}")

    return df_filtered


def analyze_before_after(df, original_col, scrubbed_col, output_dir):
    """
    Analyze differences between original and scrubbed text.

    Args:
        df: DataFrame with original and scrubbed text
        original_col: Name of original text column
        scrubbed_col: Name of scrubbed text column
        output_dir: Directory to save analysis results
    """
    logger.info("Performing before/after analysis...")

    # Find texts with the most replacements
    df["num_replacements"] = df[scrubbed_col].str.count(r"\[.*?\]")

    # Get top texts with most replacements
    top_replacements = df.nlargest(10, "num_replacements")[
        [original_col, scrubbed_col, "num_replacements"]
    ]

    # Save examples
    examples_path = Path(output_dir) / "scrubbing_examples.csv"
    save_dataframe(top_replacements, examples_path)
    logger.info(f"Saved scrubbing examples to {examples_path}")

    # Analyze by gender if available
    if "gender" in df.columns:
        gender_analysis = (
            df.groupby("gender")
            .agg(
                {
                    "num_replacements": ["mean", "std", "max"],
                    original_col: lambda x: x.str.len().mean(),
                    scrubbed_col: lambda x: x.str.len().mean(),
                }
            )
            .round(2)
        )

        logger.info("Scrubbing analysis by gender:")
        logger.info(f"\n{gender_analysis}")

        # Save gender analysis
        analysis_path = Path(output_dir) / "gender_scrubbing_analysis.csv"
        save_dataframe(gender_analysis.reset_index(), analysis_path)


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info(f"Starting text scrubbing pipeline with arguments: {vars(args)}")

        # Validate inputs
        input_path, output_path = validate_inputs(args)

        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_dataframe(input_path)

        # Check if text column exists
        if args.text_column not in df.columns:
            raise ValueError(
                f"Text column '{args.text_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # Filter data if needed
        df_filtered = filter_data(df, args.text_column, args.filter_gender)

        if len(df_filtered) == 0:
            raise ValueError("No valid texts found for scrubbing after filtering")

        # Initialize scrubber
        logger.info("Initializing text scrubber")
        scrubber = TextScrubber()

        # Perform scrubbing
        logger.info(f"Starting text scrubbing for {len(df_filtered)} texts")

        scrubbed_df = scrubber.scrub_dataframe(
            df_filtered, text_column=args.text_column, output_column=args.output_column
        )

        # Save results
        save_dataframe(scrubbed_df, output_path)
        logger.info(f"Saved scrubbed data to {output_path}")

        # Perform impact analysis
        if args.analyze:
            logger.info("Analyzing scrubbing impact...")

            stats = scrubber.analyze_scrubbing_impact(
                scrubbed_df,
                original_column=args.text_column,
                scrubbed_column=args.output_column,
            )

            logger.info("Scrubbing impact analysis:")
            for key, value in stats.items():
                if key == "placeholder_counts":
                    logger.info(f"  {key}:")
                    for placeholder, count in value.items():
                        logger.info(f"    {placeholder}: {count}")
                else:
                    logger.info(f"  {key}: {value}")

            # Save analysis results
            analysis_path = output_path.parent / f"{output_path.stem}_analysis.json"
            save_json(stats, analysis_path)
            logger.info(f"Saved analysis results to {analysis_path}")

            # Detailed before/after analysis
            analyze_before_after(
                scrubbed_df, args.text_column, args.output_column, output_path.parent
            )

        # Perform validation if requested
        if args.validate:
            logger.info(f"Starting validation with {args.validation_size} samples")

            issues = scrubber.validate_scrubbing(
                scrubbed_df,
                scrubbed_column=args.output_column,
                sample_size=args.validation_size,
            )

            if issues:
                logger.warning(f"Found {len(issues)} potential scrubbing issues:")
                for i, issue in enumerate(issues[:5]):  # Show first 5
                    logger.warning(f"  Issue {i + 1}: {issue}")

                # Save validation issues
                validation_path = (
                    output_path.parent / f"{output_path.stem}_validation_issues.json"
                )
                save_json(issues, validation_path)
                logger.info(f"Saved validation issues to {validation_path}")
            else:
                logger.info("No scrubbing issues found in validation sample")

        # Print summary
        logger.info("Text scrubbing summary:")
        if "gender" in scrubbed_df.columns:
            gender_counts = scrubbed_df["gender"].value_counts()
            for gender, count in gender_counts.items():
                logger.info(f"  {gender}: {count} texts scrubbed")

        logger.info("Text scrubbing pipeline completed successfully!")

        # Provide next steps guidance
        logger.info("\nNext steps:")
        logger.info(f"1. Review the scrubbed data in {output_path}")
        logger.info("2. Generate embeddings: python scripts/03_generate_embeddings.py")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
