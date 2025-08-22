#!/usr/bin/env python3
"""
Stage 4: Train second-stage ML models (paper-limited)

Implements only the models specified in the paper: Lasso Logistic, Random Forest,
MLP, LightGBM (if available), XGBoost (if available).
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.classifiers import EmbeddingClassifier
from src.utils.file_io import ensure_directory
from src.utils.file_io import load_dataframe
from src.utils.file_io import save_json
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train second-stage ML models on embeddings (paper set)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--embedding-dir",
        type=str,
        required=True,
        help="Directory containing embedding CSV files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained models and results",
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="gender",
        help="Target column name",
    )

    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="Filter out NEUTRAL rows to train a binary FEMALE vs MALE classifier",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available embedding CSVs in the directory",
    )

    parser.add_argument(
        "--embedding-file",
        type=str,
        help="Specific embedding CSV file to train on (overrides --compare)",
    )

    return parser.parse_args()


def main() -> None:
    try:
        args = parse_arguments()
        embedding_dir = Path(args.embedding_dir)
        output_dir = Path(args.output_dir)
        ensure_directory(output_dir)

        classifier = EmbeddingClassifier()

        results_path = output_dir / "training_results.json"

        if args.embedding_file:
            df = load_dataframe(embedding_dir / args.embedding_file)
            if args.binary_only:
                df = df[df[args.target_column] != "NEUTRAL"].reset_index(drop=True)
            embedding_name = Path(args.embedding_file).stem
            embedding_cols = [c for c in df.columns if "embed_" in c]
            X_train, X_test, y_train, y_test = classifier.prepare_data(
                df, embedding_cols, target_column=args.target_column
            )
            classifier.train_models(X_train, y_train)
            eval_results = classifier.evaluate_models(
                X_test, y_test, detailed_report=True
            )
            save_json({embedding_name: eval_results}, results_path)
            classifier.save_models(str(output_dir / embedding_name))
            logger.info(
                f"Saved models and results for {embedding_name} to {output_dir}"
            )
        else:
            # Compare across all embedding CSVs
            datasets = {}
            for f in embedding_dir.glob("*.csv"):
                try:
                    df = load_dataframe(f)
                    if args.binary_only:
                        if args.target_column not in df.columns:
                            continue
                        df = df[df[args.target_column] != "NEUTRAL"].reset_index(
                            drop=True
                        )
                    datasets[f.stem] = df
                except Exception as e:
                    logger.warning(f"Skipping {f}: {e}")

            comparison = classifier.compare_embeddings(
                datasets, target_column=args.target_column
            )
            save_json(comparison, results_path)
            classifier.save_models(str(output_dir / "best_models"))
            logger.info(f"Saved comparison results and models to {output_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
