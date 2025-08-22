#!/usr/bin/env python3
"""
Stage 3: Embedding Generation

Generates embeddings from scrubbed texts using selected methods.
Defaults to OpenAI embeddings; BERT cls/mean available as options.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.bert_embeddings import BertEmbedder
from src.embeddings.openai_embeddings import OpenAIEmbedder
from src.utils.file_io import ensure_directory
from src.utils.file_io import load_dataframe
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text embeddings from scrubbed posts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with scrubbed posts",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated embeddings",
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="scrubbed_text",
        help="Text column to embed",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["openai"],
        choices=["openai", "bert_cls", "bert_mean"],
        help="Embedding methods to run",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Optional: number of rows to process (for quick runs)",
    )

    return parser.parse_args()


def main() -> None:
    try:
        args = parse_arguments()
        logger.info(f"Starting embedding generation with arguments: {vars(args)}")

        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        ensure_directory(output_dir)

        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_dataframe(input_path)

        if args.text_column not in df.columns:
            raise ValueError(
                f"Text column '{args.text_column}' not found. Available: {list(df.columns)}"
            )

        # Optional sampling
        if args.sample_size and len(df) > args.sample_size:
            df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled to {args.sample_size} rows for embedding generation")

        # Methods
        for method in args.methods:
            if method == "openai":
                logger.info("Running OpenAI embeddings...")
                embedder = OpenAIEmbedder()
                out_path = output_dir / "openai_embeddings.csv"
                result = embedder.embed_dataframe(
                    df=df,
                    text_column=args.text_column,
                    output_prefix="openai",
                    save_path=str(out_path),
                    resume_from_checkpoint=False,
                )
                logger.info(f"Saved OpenAI embeddings to {out_path}")

            elif method == "bert_cls":
                logger.info("Running BERT CLS embeddings...")
                embedder = BertEmbedder()
                out_path = output_dir / "bert_cls_embeddings.csv"
                result = embedder.embed_dataframe(
                    df=df,
                    text_column=args.text_column,
                    strategy="cls",
                    output_prefix="bert_cls",
                    save_path=str(out_path),
                )
                logger.info(f"Saved BERT CLS embeddings to {out_path}")

            elif method == "bert_mean":
                logger.info("Running BERT mean embeddings...")
                embedder = BertEmbedder()
                out_path = output_dir / "bert_mean_embeddings.csv"
                result = embedder.embed_dataframe(
                    df=df,
                    text_column=args.text_column,
                    strategy="mean",
                    output_prefix="bert_mean",
                    save_path=str(out_path),
                )
                logger.info(f"Saved BERT mean embeddings to {out_path}")

        logger.info("Embedding generation completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Embedding generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
