"""
Comprehensive unit tests for classification models.

Tests cover model initialization, data preparation, training, evaluation,
hyperparameter tuning, model saving/loading, and error handling.
Demonstrates testing of ML pipelines with mocked dependencies.
"""

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling.classifiers import EmbeddingClassifier


class TestEmbeddingClassifier:
    """Test suite for EmbeddingClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create an EmbeddingClassifier instance."""
        return EmbeddingClassifier(random_state=42)

    @pytest.fixture
    def sample_data(self):
        """Create sample embedding data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        # Create synthetic embeddings
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(["FEMALE", "MALE", "NEUTRAL"], size=n_samples)

        # Create DataFrame with embedding columns
        embedding_cols = [f"embed_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=embedding_cols)
        df["gender"] = y

        return df, embedding_cols

    def test_initialization(self, classifier):
        """Test proper initialization of EmbeddingClassifier."""
        assert classifier.random_state == 42
        assert classifier.test_size == 0.2
        assert "Lasso Logistic" in classifier.models
        assert "Random Forest" in classifier.models
        assert isinstance(classifier.models["Lasso Logistic"], LogisticRegression)
        assert isinstance(classifier.models["Random Forest"], RandomForestClassifier)

    def test_model_configurations(self, classifier):
        """Test that models are configured correctly."""
        # Test Lasso Logistic configuration
        lasso = classifier.models["Lasso Logistic"]
        assert lasso.penalty == "l1"
        assert lasso.C == 1000.0
        assert lasso.solver == "saga"
        assert lasso.random_state == 42

        # Test Random Forest configuration
        rf = classifier.models["Random Forest"]
        assert rf.n_estimators == 300
        assert rf.random_state == 42

    def test_prepare_data(self, classifier, sample_data):
        """Test data preparation for classification."""
        df, embedding_cols = sample_data

        X_train, X_test, y_train, y_test = classifier.prepare_data(
            df, embedding_cols, target_column="gender"
        )

        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == len(df)
        assert X_train.shape[1] == len(embedding_cols)

        # Check that data is scaled
        assert np.abs(X_train.mean()) < 0.1  # Should be close to 0
        assert np.abs(X_train.std() - 1.0) < 0.3  # Should be close to 1

        # Check that labels are encoded
        assert y_train.dtype in [np.int32, np.int64]
        assert y_test.dtype in [np.int32, np.int64]

    def test_prepare_data_with_nan(self, classifier):
        """Test data preparation handles NaN values."""
        df = pd.DataFrame(
            {
                "embed_0": [1.0, np.nan, 3.0],
                "embed_1": [4.0, 5.0, np.nan],
                "gender": ["FEMALE", "MALE", "NEUTRAL"],
            }
        )

        with patch("src.modeling.classifiers.logger") as mock_logger:
            X_train, X_test, y_train, y_test = classifier.prepare_data(
                df, ["embed_0", "embed_1"]
            )

        # Check that NaN warning was logged
        mock_logger.warning.assert_called()

        # Check that NaN values were filled
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_train_models(self, classifier, sample_data):
        """Test model training."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        training_results = classifier.train_models(
            X_train, y_train, use_cross_validation=False
        )

        # Check that models were trained
        assert len(classifier.trained_models) > 0
        assert "Lasso Logistic" in training_results
        assert "Random Forest" in training_results

        # Check training results structure
        for model_name, results in training_results.items():
            if "error" not in results:
                assert "training_samples" in results

    def test_train_models_with_cross_validation(self, classifier, sample_data):
        """Test model training with cross-validation."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        training_results = classifier.train_models(
            X_train, y_train, use_cross_validation=True, cv_folds=3
        )

        # Check CV results
        for model_name, results in training_results.items():
            if "error" not in results and model_name != "MLP_Torch":
                assert "cv_accuracy_mean" in results
                assert "cv_accuracy_std" in results
                assert results["cv_accuracy_mean"] is not None

    def test_evaluate_models(self, classifier, sample_data):
        """Test model evaluation."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        # Train models first
        classifier.train_models(X_train, y_train, use_cross_validation=False)

        # Evaluate models
        evaluation_results = classifier.evaluate_models(
            X_test, y_test, detailed_report=True
        )

        # Check evaluation results
        assert len(evaluation_results) > 0

        for model_name, results in evaluation_results.items():
            if "error" not in results:
                assert "accuracy" in results
                assert 0 <= results["accuracy"] <= 1
                assert "test_samples" in results
                assert "predictions" in results
                assert "classification_report" in results
                assert "confusion_matrix" in results

    def test_evaluate_models_without_detailed_report(self, classifier, sample_data):
        """Test model evaluation without detailed report."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        classifier.train_models(X_train, y_train, use_cross_validation=False)
        evaluation_results = classifier.evaluate_models(
            X_test, y_test, detailed_report=False
        )

        for model_name, results in evaluation_results.items():
            if "error" not in results:
                assert "accuracy" in results
                assert "classification_report" not in results
                assert "confusion_matrix" not in results

    def test_torch_mlp_training(self, classifier):
        """Test PyTorch MLP model training."""
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
            epochs=2,
            batch_size=8,
        )

        assert isinstance(model, nn.Module)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(5, 10)
            output = model(test_input)
            assert output.shape == (5, 2)

    def test_compare_embeddings(self, classifier):
        """Test comparison of different embedding methods."""
        np.random.seed(42)

        # Create two different embedding datasets
        embedding_datasets = {}

        for method in ["bert", "openai"]:
            n_samples = 100
            n_features = 50 if method == "bert" else 30

            X = np.random.randn(n_samples, n_features)
            embedding_cols = [f"embed_{i}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=embedding_cols)
            df["gender"] = np.random.choice(["FEMALE", "MALE"], size=n_samples)

            embedding_datasets[method] = df

        comparison_results = classifier.compare_embeddings(embedding_datasets)

        assert len(comparison_results) == 2
        assert "bert" in comparison_results
        assert "openai" in comparison_results

        for method, results in comparison_results.items():
            if "error" not in results:
                assert "best_model" in results
                assert "best_accuracy" in results
                assert "embedding_dimension" in results
                assert "model_results" in results

    def test_hyperparameter_tuning(self, classifier, sample_data):
        """Test hyperparameter tuning functionality."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        with patch("src.modeling.classifiers.GridSearchCV") as mock_grid:
            mock_grid_instance = Mock()
            mock_grid_instance.fit = Mock()
            mock_grid_instance.best_params_ = {"C": 1.0}
            mock_grid_instance.best_score_ = 0.85
            mock_grid_instance.best_estimator_ = Mock()
            mock_grid_instance.cv_results_ = {}
            mock_grid.return_value = mock_grid_instance

            results = classifier.hyperparameter_tuning(
                X_train, y_train, model_name="Logistic Regression", cv_folds=3
            )

        assert "best_params" in results
        assert "best_score" in results
        assert results["best_score"] == 0.85

    def test_hyperparameter_tuning_invalid_model(self, classifier, sample_data):
        """Test hyperparameter tuning with invalid model name."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        with pytest.raises(ValueError, match="Model InvalidModel not available"):
            classifier.hyperparameter_tuning(
                X_train, y_train, model_name="InvalidModel"
            )

    def test_save_and_load_models(self, classifier, sample_data, tmp_path):
        """Test model saving and loading."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        # Train models
        classifier.train_models(X_train, y_train, use_cross_validation=False)

        # Save models
        save_dir = tmp_path / "models"
        classifier.save_models(str(save_dir))

        # Check that files were created
        assert (save_dir / "scaler.pkl").exists()
        assert (save_dir / "label_encoder.pkl").exists()
        assert (save_dir / "training_results.pkl").exists()

        # Create new classifier and load models
        new_classifier = EmbeddingClassifier()
        new_classifier.load_models(str(save_dir))

        # Check that models were loaded
        assert len(new_classifier.trained_models) > 0
        assert new_classifier.scaler is not None
        assert new_classifier.label_encoder is not None

    def test_load_models_missing_directory(self, classifier):
        """Test loading models from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            classifier.load_models("/non/existent/path")

    def test_error_handling_in_training(self, classifier, sample_data):
        """Test error handling during model training."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        # Add a mock model that raises an error
        classifier.models["ErrorModel"] = Mock(side_effect=Exception("Training failed"))

        with patch("src.modeling.classifiers.logger") as mock_logger:
            training_results = classifier.train_models(X_train, y_train)

        # Check that error was logged
        assert any(
            "Error training ErrorModel" in str(call)
            for call in mock_logger.error.call_args_list
        )

        # Check that error is in results
        assert "error" in training_results.get("ErrorModel", {})

    def test_error_handling_in_evaluation(self, classifier, sample_data):
        """Test error handling during model evaluation."""
        df, embedding_cols = sample_data
        X_train, X_test, y_train, y_test = classifier.prepare_data(df, embedding_cols)

        # Add a mock trained model that raises an error
        classifier.trained_models["ErrorModel"] = Mock(
            predict=Mock(side_effect=Exception("Prediction failed"))
        )

        with patch("src.modeling.classifiers.logger") as mock_logger:
            evaluation_results = classifier.evaluate_models(X_test, y_test)

        # Check that error was logged
        assert any(
            "Error evaluating ErrorModel" in str(call)
            for call in mock_logger.error.call_args_list
        )

        # Check that error is in results
        assert "error" in evaluation_results.get("ErrorModel", {})

    def test_torch_model_save_load(self, classifier, tmp_path):
        """Test saving and loading of PyTorch models."""
        # Create a simple torch model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        classifier.trained_models["MLP_Torch"] = model

        # Save models
        save_dir = tmp_path / "models"
        classifier.save_models(str(save_dir))

        # Check that state dict was saved
        assert (save_dir / "mlp_torch_state.pt").exists()

    def test_empty_embedding_columns(self, classifier):
        """Test handling of missing embedding columns."""
        df = pd.DataFrame({"text": ["Sample text"], "gender": ["FEMALE"]})

        embedding_datasets = {"test": df}

        with patch("src.modeling.classifiers.logger") as mock_logger:
            comparison_results = classifier.compare_embeddings(embedding_datasets)

        # Check that warning was logged
        mock_logger.warning.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
