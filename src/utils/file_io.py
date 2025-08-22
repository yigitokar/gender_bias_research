"""File I/O utilities for the gender bias research pipeline."""

import json
import pickle
from pathlib import Path
from typing import Any
from typing import Union

import numpy as np
import pandas as pd

from .logging_utils import setup_logger

logger = setup_logger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")
    return path


def save_dataframe(
    df: pd.DataFrame, filepath: Union[str, Path], file_format: str = "csv", **kwargs
) -> None:
    """
    Save a pandas DataFrame to file.

    Args:
        df: DataFrame to save
        filepath: Output file path
        file_format: File format ('csv', 'parquet', 'pickle')
        **kwargs: Additional arguments passed to the save method
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    try:
        if file_format.lower() == "csv":
            df.to_csv(filepath, index=False, **kwargs)
        elif file_format.lower() == "parquet":
            df.to_parquet(filepath, **kwargs)
        elif file_format.lower() == "pickle":
            df.to_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Saved DataFrame ({df.shape}) to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filepath}: {e}")
        raise


def load_dataframe(
    filepath: Union[str, Path], file_format: str = None, **kwargs
) -> pd.DataFrame:
    """
    Load a pandas DataFrame from file.

    Args:
        filepath: Input file path
        file_format: File format (auto-detected from extension if None)
        **kwargs: Additional arguments passed to the load method

    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect format from extension if not specified
    if file_format is None:
        file_format = filepath.suffix.lower().lstrip(".")

    try:
        if file_format == "csv":
            df = pd.read_csv(filepath, **kwargs)
        elif file_format == "parquet":
            df = pd.read_parquet(filepath, **kwargs)
        elif file_format == "pickle":
            df = pd.read_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Loaded DataFrame ({df.shape}) from {filepath}")
        return df

    except Exception as e:
        logger.error(f"Failed to load DataFrame from {filepath}: {e}")
        raise


def save_json(data: dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    try:

        def _json_default(obj):
            # Handle NumPy scalar types and arrays
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            # Fallback to string for unknown types
            return str(obj)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        logger.info(f"Saved JSON data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def load_json(filepath: Union[str, Path]) -> dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath) as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle object to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {filepath}: {e}")
        raise


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Input file path

    Returns:
        Loaded object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded pickle object from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load pickle from {filepath}: {e}")
        raise
