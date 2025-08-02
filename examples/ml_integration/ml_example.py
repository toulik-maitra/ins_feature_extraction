#!/usr/bin/env python3
"""
ML Integration Example
======================

This example demonstrates how to use the extracted INS features for various
machine learning tasks including classification, regression, and clustering.

Usage:
    python ml_example.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           r2_score, mean_squared_error, silhouette_score)
import warnings
warnings.filterwarnings('ignore')

class INSMLAnalyzer:
    """Machine Learning analyzer for INS spectral features."""
    
    def __init__(self, features_file="comprehensive_analysis_results/features/ml_dataset_clean.csv"):
        """
        Initialize the ML analyzer.
        
        Parameters:
        -----------
        features_file : str
            Path to the features CSV file
        """
        self.features_file = features_file
        self.features_df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_features(self, target_column=None):
        """
        Load features from CSV file.
        
        Parameters:
        -----------
        target_column : str, optional
            Name of the target column for supervised learning
        """
        try:
            self.features_df = pd.read_csv(self.features_file)
            print(f"✓ Loaded {len(self.features_df)} samples with {len(self.features_df.columns)} features")
            
            if target_column and target_column in self.features_df.columns:
                self.X = self.features_df.drop(columns=[target_column])
                self.y = self.features_df[target_column]
                print(f"✓ Target column: {target_column}")
                print(f"✓ Features shape: {self.X.shape}")
            else:
                self.X = self.features_df
                print("✓ No target column specified - ready for unsupervised learning")
                
        except FileNotFoundError:
            print(f"✗ Features file not found: {self.features_file}")
            print("Please run the INS analysis first to generate features.")
            
    def prepare_features(self, remove_non_numeric=True, handle_missing='drop'):
        """
        Prepare features for ML.
        
        Parameters:
        -----------
        remove_non_numeric : bool
            Remove non-numeric columns
        handle_missing : str
            How to handle missing values ('drop', 'fill_mean', 'fill_median')
        """
        if self.X is None:
            print("✗ No features loaded. Call load_features() first.")
            return
            
        # Remove non-numeric columns
        if remove_non_numeric:
            self.X = self.X.select_dtypes(include=[np.number])
            print(f"✓ Kept {len(self.X.columns)} numeric features")
            
        # Handle missing values
        if self.X.isnull().any().any():
            missing_count = self.X.isnull().sum().sum()
            print(f"⚠ Found {missing_count} missing values")
            
            if handle_missing == 'drop':
                self.X = self.X.dropna()
                print(f"✓ Dropped rows with missing values. Remaining: {len(self.X)} samples")
            elif handle_missing == 'fill_mean':
                self.X = self.X.fillna(self.X.mean())
                print("✓ Filled missing values with mean")
            elif handle_missing == 'fill_median':
                self.X = self.X.fillna(self.X.median())
                print("✓ Filled missing values with median")
                
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        print("✓ Features scaled using StandardScaler")
        
    def classification_example(self, test_size=0.2, random_state=42):
        """Example classification workflow."""
        if self.y is None:
            print("✗ No target variable. Load features with target_column first.")
            return
            
        # Encode target if categorical
        if self.y.dtype == 'object':
            self.y_encoded = self.label_encoder.fit_transform(self.y)
            print(f"✓ Encoded target variable: {list(self.label_encoder.classes_)}")
        else:
            self.y_encoded = self.y
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_encoded, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Train multiple classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'SVM': SVC(random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state)
        }
        
        results = {}
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            
            # Train
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            
            # Evaluate
            accuracy = clf.score(X_test, y_test)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'model': clf
            }
            
            print(f"✓ {name} Accuracy: {accuracy:.3f}")
            
            # Feature importance (for Random Forest)
            if name == 'Random Forest':
                importances = clf.feature_importances_
                feature_names = self.X.columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(importance_df.head(10))
                
        return results
    
    def regression_example(self, test_size=0.2, random_state=42):
        """Example regression workflow."""
        if self.y is None:
            print("✗ No target variable. Load features with target_column first.")
            return
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Train multiple regressors
        regressors = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'SVR': SVR(),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        for name, reg in regressors.items():
            print(f"\nTraining {name}...")
            
            # Train
            reg.fit(X_train, y_train)
            
            # Predict
            y_pred = reg.predict(X_test)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'r2': r2,
                'rmse': rmse,
                'predictions': y_pred,
                'model': reg
            }
            
            print(f"✓ {name} R²: {r2:.3f}, RMSE: {rmse:.3f}")
            
        return results
    
    def clustering_example(self, n_clusters=3, random_state=42):
        """Example clustering workflow."""
        if self.X_scaled is None:
            print("✗ No features prepared. Call prepare_features() first.")
            return
            
        print(f"✓ Clustering {len(self.X_scaled)} samples into {n_clusters} clusters...")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(self.X_scaled)
        
        # Evaluate clustering
        silhouette_avg = silhouette_score(self.X_scaled, cluster_labels)
        print(f"✓ Silhouette Score: {silhouette_avg:.3f}")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                            cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'K-Means Clustering (n_clusters={n_clusters})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs('ml_integration_results', exist_ok=True)
        plt.savefig('ml_integration_results/clustering_results.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('ml_integration_results/clustering_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save clustering results
        import os
        import pandas as pd
        
        os.makedirs('ml_integration_results', exist_ok=True)
        
        # Create results dataframe
        clustering_df = pd.DataFrame({
            'sample_index': range(len(cluster_labels)),
            'cluster_label': cluster_labels,
            'pc1': X_pca[:, 0],
            'pc2': X_pca[:, 1]
        })
        
        # Add sample names if available
        if hasattr(self, 'features_df') and 'molecule_name' in self.features_df.columns:
            clustering_df['molecule_name'] = self.features_df['molecule_name'].values
        
        clustering_df.to_csv('ml_integration_results/clustering_results.csv', index=False)
        
        # Save clustering summary
        summary_df = pd.DataFrame({
            'metric': ['n_clusters', 'silhouette_score', 'n_samples'],
            'value': [n_clusters, silhouette_avg, len(cluster_labels)]
        })
        summary_df.to_csv('ml_integration_results/clustering_summary.csv', index=False)
        
        print(f"✓ Clustering results saved to ml_integration_results/")
        
        return {
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'model': kmeans,
            'pca': pca
        }
    
    def feature_analysis(self):
        """Analyze feature distributions and correlations."""
        if self.X is None:
            print("✗ No features loaded. Call load_features() first.")
            return
            
        print("Analyzing features...")
        
        # Basic statistics
        print("\nFeature Statistics:")
        print(self.X.describe())
        
        # Correlation analysis
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.X.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs('ml_integration_results', exist_ok=True)
        plt.savefig('ml_integration_results/correlation_matrix.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('ml_integration_results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature distributions
        n_features = len(self.X.columns)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3 * n_rows))
        for i, feature in enumerate(self.X.columns):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(self.X[feature], bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'{feature}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs('ml_integration_results', exist_ok=True)
        plt.savefig('ml_integration_results/feature_distributions.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('ml_integration_results/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        # Save feature analysis results
        import os
        import pandas as pd
        
        os.makedirs('ml_integration_results', exist_ok=True)
        
        # Save feature statistics
        feature_stats = self.X.describe()
        feature_stats.to_csv('ml_integration_results/feature_statistics.csv')
        
        # Save correlation matrix
        correlation_matrix.to_csv('ml_integration_results/correlation_matrix.csv')
        
        # Save highly correlated pairs
        if high_corr_pairs:
            print("\nHighly Correlated Feature Pairs (|r| > 0.8):")
            corr_pairs_df = pd.DataFrame(high_corr_pairs, columns=['feature1', 'feature2', 'correlation'])
            corr_pairs_df.to_csv('ml_integration_results/highly_correlated_pairs.csv', index=False)
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
        else:
            print("\nNo highly correlated feature pairs found.")
            # Create empty file
            pd.DataFrame(columns=['feature1', 'feature2', 'correlation']).to_csv('ml_integration_results/highly_correlated_pairs.csv', index=False)
        
        print(f"✓ Feature analysis results saved to ml_integration_results/")
    
    def hyperparameter_tuning(self, model_type='classification', test_size=0.2, random_state=42):
        """Perform hyperparameter tuning."""
        if self.y is None:
            print("✗ No target variable. Load features with target_column first.")
            return
            
        if model_type == 'classification':
            # Encode target
            if self.y.dtype == 'object':
                y_encoded = self.label_encoder.fit_transform(self.y)
            else:
                y_encoded = self.y
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, y_encoded, test_size=test_size, random_state=random_state
            )
            
            # Random Forest tuning
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            rf = RandomForestClassifier(random_state=random_state)
            rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            rf_grid.fit(X_train, y_train)
            
            print(f"✓ Best Random Forest parameters: {rf_grid.best_params_}")
            print(f"✓ Best cross-validation score: {rf_grid.best_score_:.3f}")
            print(f"✓ Test set accuracy: {rf_grid.score(X_test, y_test):.3f}")
            
            return rf_grid
            
        elif model_type == 'regression':
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y, test_size=test_size, random_state=random_state
            )
            
            # Random Forest tuning
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            rf = RandomForestRegressor(random_state=random_state)
            rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
            rf_grid.fit(X_train, y_train)
            
            print(f"✓ Best Random Forest parameters: {rf_grid.best_params_}")
            print(f"✓ Best cross-validation R²: {rf_grid.best_score_:.3f}")
            print(f"✓ Test set R²: {rf_grid.score(X_test, y_test):.3f}")
            
            return rf_grid

def main():
    """Main function demonstrating ML workflows."""
    print("="*60)
    print("INS SPECTRUM ML ANALYSIS EXAMPLE")
    print("="*60)
    
    # Initialize analyzer
    ml_analyzer = INSMLAnalyzer()
    
    # Note: This example assumes you have already run the INS analysis
    # and have a features file. If not, you'll need to create one first.
    
    print("\nNote: This example requires a features file from INS analysis.")
    print("If you don't have one, run the INS analysis first:")
    print("python src/core/batch_ml_analysis.py --directory path/to/spectra/")
    
    # Try to load features (this will fail if file doesn't exist)
    try:
        ml_analyzer.load_features()
        ml_analyzer.prepare_features()
        
        # Feature analysis
        print("\n" + "="*40)
        print("FEATURE ANALYSIS")
        print("="*40)
        ml_analyzer.feature_analysis()
        
        # Clustering example (no target needed)
        print("\n" + "="*40)
        print("CLUSTERING EXAMPLE")
        print("="*40)
        clustering_results = ml_analyzer.clustering_example(n_clusters=3)
        
        print("\n✓ ML analysis completed successfully!")
        print("\nTo run classification or regression examples,")
        print("load features with a target column:")
        print("ml_analyzer.load_features(target_column='your_target')")
        
    except (FileNotFoundError, AttributeError) as e:
        print(f"\n✗ Error: {e}")
        print("\nTo create a features file:")
        print("1. Run INS analysis on your spectra")
        print("2. Check the comprehensive_analysis_results/features/ directory")
        print("3. Ensure you have a ml_dataset.csv file")
        print("\nExample:")
        print("python src/core/batch_ml_analysis.py --directory path/to/spectra/")

if __name__ == "__main__":
    main() 