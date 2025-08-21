"""
Classification models for evaluating embedding quality through downstream performance.

This module implements various classification algorithms to evaluate how well different
embedding methods capture semantic gender information after explicit cues are removed.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Dict, Tuple, Any, Optional, List
import logging
from pathlib import Path

from ..utils.config_loader import config
from ..utils.logging_utils import setup_logger
from ..utils.file_io import save_dataframe, load_dataframe, save_pickle, load_pickle

logger = setup_logger(__name__)


class EmbeddingClassifier:
    """
    Evaluates embedding quality through downstream gender classification performance.
    
    This class trains various classifiers on embeddings to determine how well
    different embedding methods preserve semantic gender information after
    explicit gender cues have been removed from the text.
    """
    
    def __init__(self, random_state: int = None):
        """
        Initialize the embedding classifier.
        
        Args:
            random_state: Random state for reproducible results
        """
        classification_config = config.get_model_config('classification')
        self.random_state = random_state or classification_config.get('random_state', 42)
        self.test_size = classification_config.get('test_size', 0.2)
        
        # Initialize models
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Store training results
        self.trained_models = {}
        self.training_results = {}
        
        logger.info(f"Initialized EmbeddingClassifier with {len(self.models)} models")
    
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize classification models with appropriate hyperparameters.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=10
            ),
            'MLP': MLPClassifier(
                random_state=self.random_state,
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'SVM': SVC(
                random_state=self.random_state,
                kernel='rbf',
                probability=True
            )
        }
        
        return models
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        embedding_columns: List[str],
        target_column: str = 'gender'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for classification.
        
        Args:
            df: DataFrame containing embeddings and labels
            embedding_columns: List of column names containing embedding features
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Extract features and target
        X = df[embedding_columns].values
        y = df[target_column].values
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found NaN values in embeddings, filling with zeros")
            X = np.nan_to_num(X)
        
        # Encode labels if they're strings
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Prepared data: Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_cross_validation: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all classification models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_cross_validation: Whether to use cross-validation
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of training results for each model
        """
        training_results = {}
        
        logger.info(f"Training {len(self.models)} models")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Cross-validation if requested
                if use_cross_validation:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds, 
                        scoring='accuracy'
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean, cv_std = None, None
                
                training_results[name] = {
                    'cv_accuracy_mean': cv_mean,
                    'cv_accuracy_std': cv_std,
                    'training_samples': len(X_train)
                }
                
                logger.info(f"{name} training complete. CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
            
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                training_results[name] = {'error': str(e)}
        
        self.training_results = training_results
        return training_results
    
    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        detailed_report: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            detailed_report: Whether to include detailed classification reports
            
        Returns:
            Dictionary of evaluation results for each model
        """
        evaluation_results = {}
        
        logger.info(f"Evaluating {len(self.trained_models)} trained models")
        
        for name, model in self.trained_models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                results = {
                    'accuracy': accuracy,
                    'test_samples': len(X_test),
                    'predictions': y_pred.tolist(),
                }
                
                if y_prob is not None:
                    results['probabilities'] = y_prob.tolist()
                
                if detailed_report:
                    # Classification report
                    class_report = classification_report(
                        y_test, y_pred, 
                        output_dict=True,
                        zero_division=0
                    )
                    results['classification_report'] = class_report
                    
                    # Confusion matrix
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    results['confusion_matrix'] = conf_matrix.tolist()
                
                evaluation_results[name] = results
                
                logger.info(f"{name} evaluation complete. Accuracy: {accuracy:.4f}")
            
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                evaluation_results[name] = {'error': str(e)}
        
        return evaluation_results
    
    def compare_embeddings(
        self,
        embedding_datasets: Dict[str, pd.DataFrame],
        target_column: str = 'gender'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different embedding methods by training classifiers on each.
        
        Args:
            embedding_datasets: Dictionary of embedding method name to DataFrame
            target_column: Name of target column
            
        Returns:
            Dictionary of comparison results
        """
        comparison_results = {}
        
        logger.info(f"Comparing {len(embedding_datasets)} embedding methods")
        
        for embedding_name, df in embedding_datasets.items():
            logger.info(f"Evaluating {embedding_name} embeddings...")
            
            try:
                # Find embedding columns
                embedding_cols = [col for col in df.columns if 'embed_' in col]
                
                if not embedding_cols:
                    logger.warning(f"No embedding columns found for {embedding_name}")
                    continue
                
                # Prepare data
                X_train, X_test, y_train, y_test = self.prepare_data(
                    df, embedding_cols, target_column
                )
                
                # Train models
                training_results = self.train_models(X_train, y_train)
                
                # Evaluate models
                evaluation_results = self.evaluate_models(X_test, y_test, detailed_report=False)
                
                # Store best performing model's results
                best_model = max(
                    evaluation_results.keys(),
                    key=lambda k: evaluation_results[k].get('accuracy', 0)
                )
                
                comparison_results[embedding_name] = {
                    'best_model': best_model,
                    'best_accuracy': evaluation_results[best_model]['accuracy'],
                    'embedding_dimension': len(embedding_cols),
                    'model_results': {
                        model: results['accuracy'] 
                        for model, results in evaluation_results.items()
                        if 'accuracy' in results
                    }
                }
                
                logger.info(f"{embedding_name}: Best model {best_model} - "
                           f"Accuracy: {comparison_results[embedding_name]['best_accuracy']:.4f}")
            
            except Exception as e:
                logger.error(f"Error evaluating {embedding_name}: {e}")
                comparison_results[embedding_name] = {'error': str(e)}
        
        return comparison_results
    
    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = 'Logistic Regression',
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to tune
            cv_folds: Number of CV folds for grid search
            
        Returns:
            Dictionary with best parameters and performance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        # Define parameter grids
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['lbfgs', 'liblinear']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return {}
        
        try:
            # Create fresh model instance
            base_model = self.models[model_name]
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            # Update trained model with best parameters
            self.trained_models[f"{model_name}_tuned"] = grid_search.best_estimator_
            
            logger.info(f"Hyperparameter tuning complete. Best score: {grid_search.best_score_:.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            return {'error': str(e)}
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models and preprocessing objects.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save each trained model
            for name, model in self.trained_models.items():
                model_path = output_path / f"{name.replace(' ', '_').lower()}_model.pkl"
                save_pickle(model, model_path)
            
            # Save preprocessing objects
            save_pickle(self.scaler, output_path / "scaler.pkl")
            save_pickle(self.label_encoder, output_path / "label_encoder.pkl")
            
            # Save training results
            save_pickle(self.training_results, output_path / "training_results.pkl")
            
            logger.info(f"Saved {len(self.trained_models)} models to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, model_dir: str) -> None:
        """
        Load trained models and preprocessing objects.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        try:
            # Load preprocessing objects
            self.scaler = load_pickle(model_path / "scaler.pkl")
            self.label_encoder = load_pickle(model_path / "label_encoder.pkl")
            
            # Load training results
            self.training_results = load_pickle(model_path / "training_results.pkl")
            
            # Load models
            self.trained_models = {}
            for model_file in model_path.glob("*_model.pkl"):
                model_name = model_file.stem.replace('_model', '').replace('_', ' ').title()
                self.trained_models[model_name] = load_pickle(model_file)
            
            logger.info(f"Loaded {len(self.trained_models)} models from {model_path}")
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise


def main():
    """Example usage of the EmbeddingClassifier class."""
    # This would typically be run with actual embedding data
    embeddings_path = Path(config.get('data.embeddings_path'))
    models_path = Path(config.get('data.models_path'))
    
    try:
        # Initialize classifier
        classifier = EmbeddingClassifier()
        
        # Example: Load different embedding datasets for comparison
        embedding_datasets = {}
        
        # Check for existing embedding files
        for embedding_file in embeddings_path.glob("*.csv"):
            if "bert" in embedding_file.name.lower():
                df = load_dataframe(embedding_file)
                embedding_datasets[embedding_file.stem] = df
        
        if embedding_datasets:
            # Compare embedding methods
            comparison_results = classifier.compare_embeddings(embedding_datasets)
            logger.info(f"Embedding comparison results: {comparison_results}")
            
            # Save models
            classifier.save_models(models_path)
        else:
            logger.warning("No embedding datasets found for evaluation")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()