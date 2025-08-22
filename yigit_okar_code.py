"""
================================================================================
CODE SAMPLE - Rational AI Applied AI Engineer Position
Yigit Okar | okar.yigit@gmail.com | github.com/yigitokar
================================================================================
This code sample is a subset of the repository at https://github.com/yigitokar/gender_bias_research,
which I used in the third chapter of my dissertation.

This sample demonstrates:
• Tabular ML: Lasso (L1), Random Forest, XGBoost, PyTorch MLP
• LLM APIs: GPT-4o-mini with temperature=0, chain-of-thought prompting
• Production code: Error handling, logging, batch processing, progress saving
• Testing: Comprehensive unit tests with mocks, edge cases, exception handling

"""

################################################################################
# SECTION 1: TABULAR ML MODELS - Classification Pipeline
# Shows: Lasso, Random Forest, XGBoost, PyTorch MLP, train/test split, CV
################################################################################

"""
Classification models for evaluating embedding quality through downstream performance.

This module implements various classification algorithms to evaluate how well different
embedding methods capture semantic gender information after explicit cues are removed.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class EmbeddingClassifier:
    """
    Evaluates embedding quality through downstream gender classification performance.

    This class trains various classifiers on embeddings to determine how well
    different embedding methods preserve semantic gender information after
    explicit gender cues have been removed from the text.
    """

    def __init__(self, random_state: int = 42):
        """Initialize the embedding classifier."""
        self.random_state = random_state
        self.test_size = 0.2

        # Initialize models
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Store training results
        self.trained_models = {}
        self.training_results = {}

    def _initialize_models(self) -> dict[str, Any]:
        """Initialize classification models with appropriate hyperparameters."""
        models = {}

        # Lasso Logistic (L1 regularization for feature selection)
        models["Lasso Logistic"] = LogisticRegression(
            random_state=self.random_state,
            max_iter=2000,
            solver="saga",
            penalty="l1",
            C=1000.0,  # Lambda=0.001 -> C=1000 (inverse regularization)
            n_jobs=None,
            multi_class="auto",
        )

        # Random Forest (ensemble method)
        models["Random Forest"] = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
        )

        # Optional: XGBoost and LightGBM if available
        try:
            from lightgbm import LGBMClassifier

            models["LightGBM"] = LGBMClassifier(
                random_state=self.random_state,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                n_jobs=-1,
            )
        except ImportError:
            pass

        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                random_state=self.random_state,
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                eval_metric="logloss",
                n_jobs=-1,
            )
        except ImportError:
            pass

        # Custom PyTorch MLP
        models["MLP_Torch"] = "torch_mlp"  # Special identifier

        return models

    def _train_torch_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
        output_dim: int,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> nn.Module:
        """Train custom PyTorch MLP with early stopping."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class CustomMLP(nn.Module):
            """Custom MLP architecture with batch normalization and dropout."""

            def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, 512)
                self.bn1 = nn.BatchNorm1d(512)
                self.fc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.fc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)
                self.fc4 = nn.Linear(128, out_dim)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
                x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
                x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
                x = self.fc4(x)
                return x

        model = CustomMLP(input_dim, output_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )

        # Training loop with early stopping
        best_val_acc = 0.0
        best_state = None
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_pred = val_logits.argmax(dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def prepare_data(
        self,
        df: pd.DataFrame,
        embedding_columns: list[str],
        target_column: str = "gender",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for classification with scaling and encoding."""
        # Extract features and target
        X = df[embedding_columns].values
        y = df[target_column].values

        # Handle missing values
        if np.isnan(X).any():
            print("Warning: Found NaN values in embeddings, filling with zeros")
            X = np.nan_to_num(X)

        # Encode labels if they're strings
        if y.dtype == "object":
            y = self.label_encoder.fit_transform(y)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(
            f"Data prepared: Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}"
        )
        print(f"Class distribution: {np.bincount(y_train)}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_cross_validation: bool = True,
        cv_folds: int = 5,
    ) -> dict[str, dict[str, float]]:
        """Train all classification models with optional cross-validation."""
        training_results = {}

        for name, model in self.models.items():
            print(f"Training {name}...")

            try:
                if name == "MLP_Torch":
                    # Create validation split for PyTorch model
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train,
                        y_train,
                        test_size=0.2,
                        random_state=self.random_state,
                        stratify=y_train,
                    )
                    output_dim = len(np.unique(y_train))
                    torch_model = self._train_torch_mlp(
                        X_tr,
                        y_tr,
                        X_val,
                        y_val,
                        input_dim=X_train.shape[1],
                        output_dim=output_dim,
                    )
                    self.trained_models[name] = torch_model
                    cv_mean, cv_std = None, None
                else:
                    # Train sklearn models
                    model.fit(X_train, y_train)
                    self.trained_models[name] = model

                    # Cross-validation
                    if use_cross_validation:
                        cv_scores = cross_val_score(
                            model, X_train, y_train, cv=cv_folds, scoring="accuracy"
                        )
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        cv_mean, cv_std = None, None

                training_results[name] = {
                    "cv_accuracy_mean": cv_mean,
                    "cv_accuracy_std": cv_std,
                    "training_samples": len(X_train),
                }

                if cv_mean:
                    print(f"{name}: CV accuracy {cv_mean:.4f} ± {cv_std:.4f}")

            except Exception as e:
                print(f"Error training {name}: {e}")
                training_results[name] = {"error": str(e)}

        self.training_results = training_results
        return training_results

    def evaluate_models(
        self, X_test: np.ndarray, y_test: np.ndarray, detailed_report: bool = True
    ) -> dict[str, dict[str, Any]]:
        """Evaluate trained models on test data."""
        evaluation_results = {}

        for name, model in self.trained_models.items():
            try:
                if name == "MLP_Torch":
                    # PyTorch model inference
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                    model.eval()
                    with torch.no_grad():
                        logits = model(X_t).cpu().numpy()
                    y_pred = logits.argmax(axis=1)
                    y_prob = None
                else:
                    # Sklearn model prediction
                    y_pred = model.predict(X_test)
                    y_prob = (
                        model.predict_proba(X_test)
                        if hasattr(model, "predict_proba")
                        else None
                    )

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                results = {
                    "accuracy": accuracy,
                    "test_samples": len(X_test),
                    "predictions": y_pred.tolist(),
                }

                if detailed_report:
                    results["classification_report"] = classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )
                    results["confusion_matrix"] = confusion_matrix(
                        y_test, y_pred
                    ).tolist()

                evaluation_results[name] = results
                print(f"{name}: Test accuracy {accuracy:.4f}")

            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                evaluation_results[name] = {"error": str(e)}

        return evaluation_results


################################################################################
# SECTION 2: LLM API INTEGRATION - Gender Labeling with GPT-4o-mini
# Shows: OpenAI API, chain-of-thought prompting, batch processing, error recovery
################################################################################

"""
Gender labeling pipeline using GPT-4o-mini for forum post classification.

This module implements the auto-labeling system that tags forum posts as FEMALE, MALE,
or NEUTRAL using OpenAI's GPT-4o-mini model with chain-of-thought verification.
"""

import json
import time
from pathlib import Path

import openai
import pandas as pd


class GenderLabeler:
    """
    Handles gender labeling of forum posts using GPT-4o-mini.

    Uses temperature=0 and explicit prompting with examples and chain-of-thought
    verification to minimize hallucinations. Requires both reasoning and answer
    fields in the JSON response.
    """

    def __init__(self, api_key: str = None):
        """Initialize the gender labeler with OpenAI client."""
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"
        self.temperature = 0  # Deterministic outputs
        self.max_tokens = 150

    def _create_labeling_prompt(self, text: str) -> str:
        """Create chain-of-thought prompt for gender labeling."""
        prompt = f"""
You are an expert at identifying who is being talked about in a forum post.
Determine whether the MAIN SUBJECT being discussed is FEMALE, MALE, or NEUTRAL
(cannot be determined or no specific person is referenced).

Important instructions:
- Focus on the person(s) being discussed, NOT the author's own gender.
- Use cues about the subject: pronouns (he/him/his, she/her/hers), kinship terms
  (husband/wife, brother/sister, son/daughter), gendered titles (Mr./Ms.), and
  gendered nouns (boy/girl, man/woman, lady/gentleman), or proper names.
- If multiple people of different genders are referenced without a clear primary
  subject, or if gender cannot be determined, answer NEUTRAL.

Text to analyze: "{text}"

Please respond with a JSON object containing:
1. "reasoning": Your step-by-step analysis explaining the indicators you found
2. "answer": One of "FEMALE", "MALE", or "NEUTRAL"

Example response format:
{{"reasoning": "The post discusses 'my wife' and 'she', indicating female subject.", "answer": "FEMALE"}}

JSON Response:"""
        return prompt

    def _parse_response(self, response_text: str) -> tuple[str, str]:
        """Parse JSON response with error handling."""
        try:
            # Clean up response text
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "")

            parsed = json.loads(response_text)
            reasoning = parsed.get("reasoning", "")
            answer = parsed.get("answer", "").upper()

            if answer not in ["FEMALE", "MALE", "NEUTRAL"]:
                raise ValueError(f"Invalid answer: {answer}")

            return reasoning, answer

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse response: {response_text[:100]}... Error: {e}")
            return "", "NEUTRAL"

    def label_single_text(self, text: str) -> dict[str, str]:
        """Label a single text for gender with comprehensive error handling."""
        if not text or pd.isna(text):
            return {
                "gender": "NEUTRAL",
                "reasoning": "Empty text",
                "confidence": "high",
            }

        try:
            prompt = self._create_labeling_prompt(text)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            response_text = response.choices[0].message.content
            reasoning, answer = self._parse_response(response_text)

            return {
                "gender": answer,
                "reasoning": reasoning,
                "confidence": "high" if reasoning else "low",
            }

        except Exception as e:
            print(f"Error labeling text: {e}")
            return {
                "gender": "NEUTRAL",
                "reasoning": f"Error: {e}",
                "confidence": "low",
            }

    def label_batch(
        self,
        texts: list[str],
        batch_size: int = 1000,
        save_progress: bool = True,
        output_path: str = None,
    ) -> pd.DataFrame:
        """Label batch of texts with progress saving and rate limiting."""
        results = []
        total_texts = len(texts)

        print(f"Starting batch labeling of {total_texts} texts")

        for i, text in enumerate(texts):
            try:
                result = self.label_single_text(text)
                result["text"] = text
                result["index"] = i
                results.append(result)

                # Progress logging
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / total_texts * 100
                    print(f"Processed {i + 1}/{total_texts} texts ({progress:.1f}%)")

                # Save progress periodically
                if save_progress and (i + 1) % batch_size == 0 and output_path:
                    df_progress = pd.DataFrame(results)
                    df_progress.to_csv(output_path, index=False)
                    print(f"Saved progress after processing {i + 1} texts")

                # Rate limiting for API
                time.sleep(0.1)

            except Exception as e:
                print(f"Error processing text {i}: {e}")
                results.append(
                    {
                        "text": text,
                        "index": i,
                        "gender": "NEUTRAL",
                        "reasoning": f"Processing error: {e}",
                        "confidence": "low",
                    }
                )

        df_results = pd.DataFrame(results)

        if save_progress and output_path:
            df_results.to_csv(output_path, index=False)

        gender_counts = df_results["gender"].value_counts().to_dict()
        print(f"Completed batch labeling. Results: {gender_counts}")

        return df_results


################################################################################
# SECTION 3: PRODUCTION ERROR HANDLING - File I/O with Recovery
# Shows: Comprehensive exception handling, type safety, automatic recovery
################################################################################

"""File I/O with comprehensive error handling."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_directory(path: Path) -> None:
    """Ensure directory exists, create if it doesn't."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create directory {path}: {e}") from e


def save_json(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save dictionary to JSON with NumPy type handling.

    Production features:
    - Handles NumPy types (int64, float64, ndarray)
    - Creates parent directories automatically
    - Comprehensive error messages
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    class NumpyEncoder(json.JSONEncoder):
        """JSON encoder that handles NumPy types."""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    except Exception as e:
        raise OSError(f"Failed to save JSON to {file_path}: {e}") from e


def load_json(file_path: str | Path) -> dict[str, Any]:
    """Load JSON with error handling."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e
    except Exception as e:
        raise OSError(f"Failed to load JSON from {file_path}: {e}") from e


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Load DataFrame with automatic format detection and error handling."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    except Exception as e:
        raise OSError(f"Failed to load dataframe from {file_path}: {e}") from e


def save_dataframe(df: pd.DataFrame, file_path: str | Path, **kwargs) -> None:
    """Save DataFrame with format auto-detection."""
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    try:
        if file_path.suffix == ".csv":
            df.to_csv(file_path, index=False, **kwargs)
        elif file_path.suffix == ".parquet":
            df.to_parquet(file_path, index=False, **kwargs)
        elif file_path.suffix == ".json":
            df.to_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
    except Exception as e:
        raise OSError(f"Failed to save dataframe to {file_path}: {e}") from e


################################################################################
# SECTION 4: COMPREHENSIVE UNIT TESTS - LLM Integration Testing
# Shows: Mocking, edge cases, exception handling, batch processing validation
################################################################################

"""Unit tests demonstrating mocking, edge cases, and exception handling."""

import json
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest


class TestGenderLabeler:
    """Comprehensive test suite for LLM integration."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing without API calls."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps(
            {
                "reasoning": "The text mentions 'she' and 'her', indicating female.",
                "answer": "FEMALE",
            }
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def labeler(self, mock_openai_client):
        """Create labeler with mocked client."""
        with patch("openai.OpenAI", return_value=mock_openai_client):
            return GenderLabeler(api_key="test_key")

    def test_label_empty_text(self, labeler):
        """Test handling of empty/None/NaN inputs."""
        assert labeler.label_single_text("")["gender"] == "NEUTRAL"
        assert labeler.label_single_text(None)["gender"] == "NEUTRAL"
        assert labeler.label_single_text(pd.NA)["gender"] == "NEUTRAL"

    def test_api_error_handling(self, labeler, mock_openai_client):
        """Test graceful handling of API failures."""
        mock_openai_client.chat.completions.create.side_effect = Exception(
            "Rate limit exceeded"
        )

        result = labeler.label_single_text("Test text")
        assert result["gender"] == "NEUTRAL"
        assert "Rate limit exceeded" in result["reasoning"]
        assert result["confidence"] == "low"

    def test_malformed_json_response(self, labeler, mock_openai_client):
        """Test handling of invalid JSON from API."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "Invalid JSON"

        result = labeler.label_single_text("Test")
        assert result["gender"] == "NEUTRAL"  # Fallback to safe default
        assert result["confidence"] == "low"

    def test_json_with_code_blocks(self, labeler, mock_openai_client):
        """Test parsing JSON wrapped in markdown code blocks."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = """```json
        {
            "reasoning": "Subject is identified as female.",
            "answer": "FEMALE"
        }
        ```"""

        result = labeler.label_single_text("She is talented.")
        assert result["gender"] == "FEMALE"
        assert result["confidence"] == "high"

    def test_invalid_gender_answer(self, labeler, mock_openai_client):
        """Test handling of invalid gender values in response."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(
            {
                "reasoning": "Some reasoning",
                "answer": "UNKNOWN",  # Invalid value
            }
        )

        result = labeler.label_single_text("Test text.")
        assert result["gender"] == "NEUTRAL"  # Fallback for invalid answers
        assert result["confidence"] == "low"

    def test_batch_with_mixed_failures(self, labeler, mock_openai_client):
        """Test batch processing handles partial failures gracefully."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Simulate: success, failure, success
        side_effects = [
            Mock(
                choices=[
                    Mock(message=Mock(content='{"answer":"FEMALE","reasoning":"R1"}'))
                ]
            ),
            Exception("API Error"),
            Mock(
                choices=[
                    Mock(message=Mock(content='{"answer":"MALE","reasoning":"R3"}'))
                ]
            ),
        ]
        mock_openai_client.chat.completions.create.side_effect = side_effects

        with patch("time.sleep"):  # Speed up test
            results = labeler.label_batch(texts, save_progress=False)

        assert len(results) == 3
        assert results.iloc[0]["gender"] == "FEMALE"
        assert results.iloc[1]["gender"] == "NEUTRAL"  # Error fallback
        assert results.iloc[2]["gender"] == "MALE"

    def test_prompt_creation(self, labeler):
        """Test that prompts contain required elements."""
        text = "She told her mother about the job."
        prompt = labeler._create_labeling_prompt(text)

        assert text in prompt
        assert "FEMALE" in prompt
        assert "MALE" in prompt
        assert "NEUTRAL" in prompt
        assert "reasoning" in prompt
        assert "answer" in prompt

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_rate_limiting(self, mock_sleep, labeler):
        """Test that rate limiting is applied during batch processing."""
        texts = ["Text 1", "Text 2"]

        labeler.label_batch(texts, save_progress=False)

        # Verify sleep was called for rate limiting
        assert mock_sleep.call_count >= len(texts)


################################################################################
# SECTION 5: ML PIPELINE TESTS - Model Training and Evaluation
# Shows: Synthetic data generation, model testing, error injection
################################################################################

"""Tests for ML models including PyTorch, showing synthetic data generation."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TestEmbeddingClassifier:
    """Test suite for tabular ML models."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return EmbeddingClassifier(random_state=42)

    @pytest.fixture
    def sample_data(self):
        """Generate synthetic embedding data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        # Create synthetic embeddings with some signal
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(["FEMALE", "MALE", "NEUTRAL"], size=n_samples)

        # Create DataFrame with embedding columns
        embedding_cols = [f"embed_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=embedding_cols)
        df["gender"] = y

        return df, embedding_cols

    def test_data_preparation_with_scaling(self, classifier, sample_data):
        """Test data preparation includes proper scaling and encoding."""
        df, embedding_cols = sample_data

        X_train, X_test, y_train, y_test = classifier.prepare_data(
            df, embedding_cols, target_column="gender"
        )

        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == len(df)
        assert X_train.shape[1] == len(embedding_cols)

        # Check that data is scaled (mean ~0, std ~1)
        assert abs(X_train.mean()) < 0.1
        assert abs(X_train.std() - 1.0) < 0.3

        # Check that labels are encoded as integers
        assert y_train.dtype in [np.int32, np.int64]
        assert y_test.dtype in [np.int32, np.int64]

    def test_torch_mlp_training_and_inference(self, classifier):
        """Test PyTorch MLP model training and inference."""
        np.random.seed(42)
        X_train = np.random.randn(50, 10).astype(np.float32)
        y_train = np.random.choice([0, 1], size=50)
        X_val = np.random.randn(10, 10).astype(np.float32)
        y_val = np.random.choice([0, 1], size=10)

        model = classifier._train_torch_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            input_dim=10,
            output_dim=2,
            epochs=2,  # Quick training for test
        )

        # Verify model is a PyTorch module
        assert isinstance(model, nn.Module)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(5, 10)
            output = model(test_input)
            assert output.shape == (5, 2)

    def test_model_evaluation_with_metrics(self, classifier, sample_data):
        """Test comprehensive model evaluation with all metrics."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        # Train models (skip CV for speed)
        classifier.train_models(X_train, y_train, use_cross_validation=False)

        # Evaluate with detailed reports
        evaluation_results = classifier.evaluate_models(
            X_test, y_test, detailed_report=True
        )

        # Check that we have results for each model
        assert len(evaluation_results) > 0

        for model_name, results in evaluation_results.items():
            if "error" not in results:
                # Check required metrics
                assert 0 <= results["accuracy"] <= 1
                assert "classification_report" in results
                assert "confusion_matrix" in results
                assert len(results["predictions"]) == len(X_test)

    def test_data_with_nan_handling(self, classifier):
        """Test that NaN values in data are handled properly."""
        df = pd.DataFrame(
            {
                "embed_0": [1.0, np.nan, 3.0, 4.0],
                "embed_1": [4.0, 5.0, np.nan, 7.0],
                "gender": ["FEMALE", "MALE", "NEUTRAL", "FEMALE"],
            }
        )

        X_train, X_test, y_train, y_test = classifier.prepare_data(
            df, ["embed_0", "embed_1"]
        )

        # Check that NaN values were handled (filled with zeros)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_error_injection_and_recovery(self, classifier, sample_data):
        """Test that pipeline handles model failures gracefully."""
        df, embedding_cols = sample_data
        X_train, _, y_train, _ = classifier.prepare_data(df, embedding_cols)

        # Inject a failing model
        classifier.models["FailingModel"] = Mock(
            fit=Mock(side_effect=Exception("Training failed"))
        )

        # Train models - should handle the failure gracefully
        results = classifier.train_models(X_train, y_train)

        # Verify error was captured but didn't crash pipeline
        assert "error" in results["FailingModel"]
        assert "Training failed" in results["FailingModel"]["error"]

        # Other models should still train successfully
        assert "Lasso Logistic" in classifier.trained_models
        assert "Random Forest" in classifier.trained_models

    def test_cross_validation_integration(self, classifier, sample_data):
        """Test cross-validation produces reasonable results."""
        df, embedding_cols = sample_data
        X_train, _, y_train, _ = classifier.prepare_data(df, embedding_cols)

        results = classifier.train_models(
            X_train, y_train, use_cross_validation=True, cv_folds=3
        )

        # Check that CV results are present for sklearn models
        for model_name, result in results.items():
            if "error" not in result and model_name != "MLP_Torch":
                assert "cv_accuracy_mean" in result
                assert "cv_accuracy_std" in result
                assert result["cv_accuracy_mean"] is not None
                assert 0 <= result["cv_accuracy_mean"] <= 1


################################################################################
# END OF CODE SAMPLE
################################################################################

"""
SUMMARY:
========

This code sample demonstrates:

1. TABULAR ML MODELS:
   - Lasso Logistic Regression with L1 regularization for feature selection
   - Random Forest ensemble method with 300 estimators
   - XGBoost/LightGBM gradient boosting (optional dependencies)
   - Custom PyTorch MLP with batch normalization and early stopping

2. LLM API INTEGRATION:
   - GPT-4o-mini API with temperature=0 for deterministic outputs
   - Chain-of-thought prompting to reduce hallucinations
   - Comprehensive JSON parsing with error recovery
   - Batch processing with progress saving and rate limiting

3. PRODUCTION FEATURES:
   - Comprehensive error handling with specific exception types
   - Automatic directory creation and file format detection
   - NumPy-compatible JSON serialization
   - Progress monitoring and logging throughout

4. TESTING EXCELLENCE:
   - Mock objects for external API dependencies
   - Edge case testing (empty, None, NaN inputs)
   - Error injection to test failure recovery
   - Synthetic data generation for reproducible tests

The pipeline processes 1M+ forum posts, achieving 79% gender classification
accuracy even after removing explicit gender cues, demonstrating the persistence
of semantic gender information in text embeddings.
"""
