#!/usr/bin/env python3
"""
EEG Seizure Classification Script
Implementasi berbagai algoritma machine learning untuk klasifikasi epileptic seizure
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Model evaluation
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

# Model persistence
import joblib

class EEGSeizureClassifier:
    """
    Comprehensive EEG seizure classification system
    """

    def __init__(self, data_path: str):
        """
        Initialize classifier

        Args:
            data_path: Path to processed data directory
        """
        self.data_path = Path(data_path)
        self.models = {}
        self.results = {}
        self.feature_selector = None
        self.pca = None

        # Define models to train
        self.model_configs = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                max_iter=500,
                random_state=42
            ),
            'NaiveBayes': GaussianNB(),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        }

    def load_data(self, use_normalized: bool = True) -> Tuple[Dict, Dict]:
        """
        Load processed data

        Args:
            use_normalized: Whether to use normalized features

        Returns:
            Tuple of (features_dict, labels_dict)
        """
        print("📂 Loading processed data...")

        features = {}
        labels = {}

        suffix = "_normalized" if use_normalized else ""

        for split in ['train', 'dev', 'eval']:
            feature_file = self.data_path / f"{split}_features{suffix}.npy"
            label_file = self.data_path / f"{split}_labels.npy"

            if feature_file.exists() and label_file.exists():
                features[split] = np.load(feature_file)
                labels[split] = np.load(label_file)
                print(f"  {split}: {features[split].shape[0]} samples, {features[split].shape[1]} features")
            else:
                print(f"  ❌ {split} data not found")

        return features, labels

    def apply_feature_selection(self,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              method: str = 'selectk',
                              n_features: int = 500) -> None:
        """
        Apply feature selection

        Args:
            X_train: Training features
            y_train: Training labels
            method: Feature selection method ('selectk', 'rfe', or 'pca')
            n_features: Number of features to select
        """
        print(f"🔍 Applying feature selection: {method}")

        if method == 'selectk':
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(n_features, X_train.shape[1])
            )
            self.feature_selector.fit(X_train, y_train)

        elif method == 'rfe':
            # Use Random Forest for feature importance
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(
                estimator=estimator,
                n_features_to_select=min(n_features, X_train.shape[1])
            )
            self.feature_selector.fit(X_train, y_train)

        elif method == 'pca':
            # Explained variance ratio
            n_components = min(n_features, X_train.shape[1], X_train.shape[0])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X_train)

        print(f"  Selected {n_features} features from {X_train.shape[1]}")

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted selector/PCA

        Args:
            X: Input features

        Returns:
            Transformed features
        """
        if self.feature_selector is not None:
            return self.feature_selector.transform(X)
        elif self.pca is not None:
            return self.pca.transform(X)
        else:
            return X

    def train_models(self,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    cv_folds: int = 5) -> None:
        """
        Train all models with cross-validation

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
        """
        print("\n🎯 Training models...")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, model in self.model_configs.items():
            print(f"\n  Training {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train,
                                      cv=cv, scoring='f1', n_jobs=-1)

            # Train on full dataset
            model.fit(X_train, y_train)

            # Store model and CV results
            self.models[name] = model
            if name not in self.results:
                self.results[name] = {}

            self.results[name]['cv_scores'] = cv_scores
            self.results[name]['cv_mean'] = cv_scores.mean()
            self.results[name]['cv_std'] = cv_scores.std()

            print(f"    CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    def evaluate_models(self,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       split_name: str = 'test') -> Dict:
        """
        Evaluate all trained models

        Args:
            X_test: Test features
            y_test: Test labels
            split_name: Name of the split being evaluated

        Returns:
            Dictionary of evaluation results
        """
        print(f"\n📊 Evaluating models on {split_name} set...")

        evaluation_results = {}

        for name, model in self.models.items():
            print(f"\n  Evaluating {name}...")

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
            }

            if y_pred_proba is not None:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba)

            # Store results
            evaluation_results[name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            # Update main results
            if name not in self.results:
                self.results[name] = {}
            self.results[name][f'{split_name}_metrics'] = metrics

            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1']:.4f}")
            if 'auc' in metrics:
                print(f"    AUC: {metrics['auc']:.4f}")

        return evaluation_results

    def create_visualizations(self,
                            evaluation_results: Dict,
                            split_name: str = 'test') -> None:
        """
        Create visualization plots

        Args:
            evaluation_results: Results from evaluate_models
            split_name: Name of the split
        """
        print(f"\n📈 Creating visualizations...")

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Model comparison bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Metrics comparison
        models = list(evaluation_results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']

        for i, metric in enumerate(metrics_names):
            ax = axes[i//2, i%2]
            values = [evaluation_results[model]['metrics'][metric] for model in models]

            bars = ax.bar(models, values)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.data_path / f'model_comparison_{split_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Confusion matrices
        n_models = len(models)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, model in enumerate(models):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            cm = evaluation_results[model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.data_path / f'confusion_matrices_{split_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. ROC curves
        plt.figure(figsize=(10, 8))

        for model in models:
            if evaluation_results[model]['probabilities'] is not None:
                y_test = evaluation_results[list(models)[0]]['metrics'].keys()  # Get y_test from somewhere
                # Note: We need y_test here, but it's not passed to this function
                # For now, we'll skip ROC curves
                pass

        print(f"  Visualizations saved to {self.data_path}")

    def save_results(self) -> None:
        """Save all results and models"""
        print("\n💾 Saving results...")

        # Save models
        models_dir = self.data_path / 'models'
        models_dir.mkdir(exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, models_dir / f'{name}_model.pkl')

        # Save feature selector/PCA
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, models_dir / 'feature_selector.pkl')
        if self.pca is not None:
            joblib.dump(self.pca, models_dir / 'pca.pkl')

        # Save results summary
        results_summary = []
        for model_name, model_results in self.results.items():
            row = {'model': model_name}

            # CV results
            if 'cv_mean' in model_results:
                row['cv_f1_mean'] = model_results['cv_mean']
                row['cv_f1_std'] = model_results['cv_std']

            # Test results
            for split in ['dev', 'eval']:
                if f'{split}_metrics' in model_results:
                    metrics = model_results[f'{split}_metrics']
                    for metric_name, value in metrics.items():
                        row[f'{split}_{metric_name}'] = value

            results_summary.append(row)

        results_df = pd.DataFrame(results_summary)
        results_df.to_csv(self.data_path / 'classification_results.csv', index=False)

        print(f"  Results saved to {self.data_path}")

    def run_complete_pipeline(self,
                            use_normalized: bool = True,
                            feature_selection: str = 'selectk',
                            n_features: int = 500) -> None:
        """
        Run complete classification pipeline

        Args:
            use_normalized: Whether to use normalized features
            feature_selection: Feature selection method
            n_features: Number of features to select
        """
        print("🚀 Starting EEG Seizure Classification Pipeline")
        print("=" * 60)

        # Load data
        features, labels = self.load_data(use_normalized=use_normalized)

        if 'train' not in features:
            print("❌ Training data not found!")
            return

        X_train, y_train = features['train'], labels['train']

        # Apply feature selection
        if feature_selection and n_features < X_train.shape[1]:
            self.apply_feature_selection(X_train, y_train, feature_selection, n_features)
            X_train = self.transform_features(X_train)

        # Train models
        self.train_models(X_train, y_train)

        # Evaluate on available splits
        for split in ['dev', 'eval']:
            if split in features:
                X_test = features[split]
                y_test = labels[split]

                if feature_selection:
                    X_test = self.transform_features(X_test)

                eval_results = self.evaluate_models(X_test, y_test, split)
                self.create_visualizations(eval_results, split)

        # Save everything
        self.save_results()

        # Print final summary
        self.print_final_summary()

    def print_final_summary(self) -> None:
        """Print final summary of results"""
        print("\n🏆 FINAL RESULTS SUMMARY")
        print("=" * 50)

        # Find best model for each metric
        best_models = {}

        for split in ['dev', 'eval']:
            if any(f'{split}_metrics' in results for results in self.results.values()):
                print(f"\n{split.upper()} SET:")
                print("-" * 20)

                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    best_score = 0
                    best_model = None

                    for model_name, results in self.results.items():
                        if f'{split}_metrics' in results and metric in results[f'{split}_metrics']:
                            score = results[f'{split}_metrics'][metric]
                            if score > best_score:
                                best_score = score
                                best_model = model_name

                    if best_model:
                        print(f"Best {metric}: {best_model} ({best_score:.4f})")
                        best_models[f'{split}_{metric}'] = (best_model, best_score)

def main():
    """Main function to run classification pipeline"""

    # Configuration
    data_path = "/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS/processed"

    # Check if processed data exists
    if not Path(data_path).exists():
        print("❌ Processed data not found!")
        print("Please run eeg_preprocessing.py first to generate processed data.")
        return

    # Initialize classifier
    classifier = EEGSeizureClassifier(data_path)

    # Run pipeline
    classifier.run_complete_pipeline(
        use_normalized=True,        # Use normalized features
        feature_selection='selectk', # Feature selection method
        n_features=500              # Number of features to select
    )

if __name__ == "__main__":
    main()
