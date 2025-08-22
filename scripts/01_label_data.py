#!/usr/bin/env python3
"""
Stage 1: Gender Labeling Pipeline

This script implements the auto-labeling system that tags ≈1M forum posts as FEMALE, MALE,
or NEUTRAL using GPT-4o-mini with temperature=0 and chain-of-thought verification.

Usage:
    python scripts/01_label_data.py --input data/raw/forum_posts.csv --output data/processed/labeled_posts.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.gender_labeling import GenderLabeler
from src.utils.file_io import ensure_directory
from src.utils.file_io import load_dataframe
from src.utils.file_io import save_dataframe
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label forum posts for gender using GPT-4o-mini",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file containing forum posts",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Path to save labeled output CSV file"
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="raw_cat",
        help="Name of column containing text to label",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of texts to process before saving progress",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of posts to process (for testing). If not specified, processes all posts.",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform validation on a sample of labels",
    )

    parser.add_argument(
        "--validation-size",
        type=int,
        default=500,
        help="Number of samples to use for validation",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing progress file"
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


def filter_data(df, text_column, sample_size=None):
    """
    Filter and prepare data for labeling.

    Args:
        df: Input DataFrame
        text_column: Name of text column
        sample_size: Optional sample size for testing

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Original dataset shape: {df.shape}")

    # Remove rows with missing text
    df_filtered = df.dropna(subset=[text_column])
    logger.info(f"After removing missing text: {df_filtered.shape}")

    # Remove empty strings
    df_filtered = df_filtered[df_filtered[text_column].str.strip() != ""]
    logger.info(f"After removing empty strings: {df_filtered.shape}")

    # Sample if requested
    if sample_size and len(df_filtered) > sample_size:
        df_filtered = df_filtered.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled to {sample_size} posts for testing")

    # Reset index to ensure alignment with labeling outputs
    df_filtered = df_filtered.reset_index(drop=True)

    return df_filtered


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info(f"Starting gender labeling pipeline with arguments: {vars(args)}")

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

        # Filter and prepare data
        df_filtered = filter_data(df, args.text_column, args.sample_size)

        if len(df_filtered) == 0:
            raise ValueError("No valid texts found for labeling after filtering")

        # Initialize labeler
        logger.info("Initializing GPT-4o-mini gender labeler")
        labeler = GenderLabeler()

        # Perform labeling
        logger.info(f"Starting labeling of {len(df_filtered)} texts")

        labeled_df = labeler.label_batch(
            texts=df_filtered[args.text_column].tolist(),
            batch_size=args.batch_size,
            save_progress=True,
            output_path=str(output_path),
        )

        # Merge with original data
        result_df = df_filtered.copy()
        result_df["gender"] = labeled_df["gender"]
        result_df["reasoning"] = labeled_df["reasoning"]
        result_df["confidence"] = labeled_df["confidence"]
        # Ensure a 'text' column exists for validation routines
        result_df["text"] = df_filtered[args.text_column]

        # Save final results
        save_dataframe(result_df, output_path)
        logger.info(f"Saved labeled data to {output_path}")

        # Print summary statistics
        gender_counts = result_df["gender"].value_counts()
        logger.info("Labeling summary:")
        for gender, count in gender_counts.items():
            percentage = count / len(result_df) * 100
            logger.info(f"  {gender}: {count} ({percentage:.1f}%)")

        # Perform validation if requested
        if args.validate:
            logger.info(f"Starting validation with {args.validation_size} samples")

            validation_results = labeler.validate_labels(
                result_df, sample_size=args.validation_size
            )

            logger.info("Validation results:")
            for key, value in validation_results.items():
                if key == "agreement_rate":
                    logger.info(f"  {key}: {value:.2%}")
                else:
                    logger.info(f"  {key}: {value}")

            # Save validation results
            validation_path = output_path.parent / f"{output_path.stem}_validation.json"
            import json

            with open(validation_path, "w") as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"Saved validation results to {validation_path}")

        logger.info("Gender labeling pipeline completed successfully!")

        # Provide next steps guidance
        logger.info("\nNext steps:")
        logger.info(f"1. Review the labeled data in {output_path}")
        logger.info("2. Run text scrubbing: python scripts/02_scrub_text.py")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
