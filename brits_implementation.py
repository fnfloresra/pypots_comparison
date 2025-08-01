# Create BRITS implementation

# ===============================================
# BRITS Implementation with PyPOTS
# ===============================================

import numpy as np
import pandas as pd
import torch
from pypots.imputation import BRITS
from pypots.nn.functional import calc_mae, calc_mse, calc_mre
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PyPOTS_BRITS_Imputer:
    def __init__(self, n_steps=30, n_features=5, rnn_hidden_size=256, 
                 epochs=100, batch_size=32, learning_rate=0.001):
        """
        BRITS (Bidirectional Recurrent Imputation for Time Series) implementation
        
        Args:
            n_steps: Number of time steps in each sequence
            n_features: Number of features (5 for your dataset)
            rnn_hidden_size: Hidden size for RNN layers
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
        """
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_data(self, csv_path):
        """
        Prepare CSV data for PyPOTS BRITS format
        
        Expected CSV format:
        - Date column (will be used as index)
        - 5 feature columns for multivariate time series
        """
        print("Loading and preparing data...")
        
        # Load CSV data
        df = pd.read_csv(csv_path)

        # Handle date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
        
        # Select numerical columns (assuming first 5 are the features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 5:
            raise ValueError(f"Expected at least 5 numerical columns, got {len(numeric_cols)}")
            
        feature_cols = numeric_cols[:5]
        df_features = df[feature_cols]
        
        # Resample to ensure daily frequency
        df_resampled = df_features.resample('D').mean()
        
        # Fill any remaining gaps with forward fill, then backward fill
        df_filled = df_resampled.ffill().bfill()
        
        print(f"Data shape after preparation: {df_filled.shape}")
        print(f"Date range: {df_filled.index.min()} to {df_filled.index.max()}")
        
        return df_filled.values, feature_cols.tolist()
    
    def create_sequences(self, data, seq_length=None):
        """
        Create overlapping sequences for BRITS training
        
        Args:
            data: Time series data (time_steps, features)
            seq_length: Length of each sequence (uses self.n_steps if None)
        """
        if seq_length is None:
            seq_length = self.n_steps
            
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        
        sequences = np.array(sequences)
        print(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        
        return sequences
    
    def introduce_missing_values(self, data, missing_rate=0.1):
        """
        Introduce missing values for testing
        
        Args:
            data: Complete data
            missing_rate: Proportion of values to make missing
        """
        data_with_missing = data.copy()
        missing_mask = np.random.random(data.shape) < missing_rate
        data_with_missing[missing_mask] = np.nan
        
        print(f"Introduced {missing_mask.sum()} missing values "
              f"({missing_mask.sum() / data.size * 100:.2f}%)")
        
        return data_with_missing, missing_mask
    
    def fit(self, X):
        """
        Fit BRITS model
        
        Args:
            X: Training data (n_samples, n_steps, n_features)
        """
        print("Initializing BRITS model...")
        
        # Normalize data
        X_reshaped = X.reshape(-1, X.shape[-1])
        # Handle NaN values during normalization
        X_reshaped_clean = np.nan_to_num(X_reshaped, nan=0.0)
        self.scaler.fit(X_reshaped_clean)
        
        X_normalized = self.scaler.transform(X_reshaped_clean)
        X_normalized = X_normalized.reshape(X.shape)
        
        # Put back NaN values
        nan_mask = np.isnan(X)
        X_normalized[nan_mask] = np.nan
        
        # Initialize BRITS model
        self.model = BRITS(
            n_steps=self.n_steps,
            n_features=self.n_features,
            rnn_hidden_size=self.rnn_hidden_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Training BRITS model...")
        print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Prepare data in PyPOTS format (dictionary with 'X' key)
        train_data = {"X": X_normalized}
        
        # Train the model
        self.model.fit(train_data)
        self.is_fitted = True
        
        print("BRITS model training completed!")
        return self
    
    def impute(self, X):
        """
        Perform imputation using trained BRITS model
        
        Args:
            X: Data with missing values (n_samples, n_steps, n_features)
            
        Returns:
            Imputed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before imputation!")
        
        print("Performing BRITS imputation...")
        
        # Normalize data
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_reshaped_clean = np.nan_to_num(X_reshaped, nan=0.0)
        X_normalized = self.scaler.transform(X_reshaped_clean)
        X_normalized = X_normalized.reshape(X.shape)
        
        # Put back NaN values
        nan_mask = np.isnan(X)
        X_normalized[nan_mask] = np.nan
        
        # Prepare data in PyPOTS format (dictionary with 'X' key)
        test_data = {"X": X_normalized}
        
        # Perform imputation
        X_imputed = self.model.impute(test_data)
        
        # Denormalize
        X_imputed_reshaped = X_imputed.reshape(-1, X_imputed.shape[-1])
        X_imputed_denorm = self.scaler.inverse_transform(X_imputed_reshaped)
        X_imputed_final = X_imputed_denorm.reshape(X_imputed.shape)
        
        print("BRITS imputation completed!")
        return X_imputed_final
    
    def evaluate_performance(self, X_true, X_imputed, missing_mask):
        """
        Evaluate BRITS imputation performance
        
        Args:
            X_true: Ground truth data
            X_imputed: Imputed data
            missing_mask: Boolean mask indicating missing values
        """
        print("Evaluating BRITS performance...")
        
        # Calculate metrics only on originally missing values
        mae = calc_mae(X_imputed[missing_mask], X_true[missing_mask])
        mse = calc_mse(X_imputed[missing_mask], X_true[missing_mask])
        mre = calc_mre(X_imputed[missing_mask], X_true[missing_mask])
        
        metrics = {
            'MAE': float(mae),
            'MSE': float(mse),
            'MRE': float(mre),
            'RMSE': float(np.sqrt(mse))
        }
        
        print("BRITS Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def plot_imputation_results(self, X_true, X_imputed, missing_mask, 
                               feature_names=None, sample_idx=0):
        """
        Plot imputation results for visualization
        
        Args:
            X_true: Ground truth data
            X_imputed: Imputed data
            missing_mask: Missing value mask
            feature_names: Names of features
            sample_idx: Sample index to plot
        """
        if feature_names is None:
            feature_names = [f'Feature_{i+1}' for i in range(self.n_features)]
        
        fig, axes = plt.subplots(self.n_features, 1, figsize=(12, 2*self.n_features))
        if self.n_features == 1:
            axes = [axes]
        
        for i in range(self.n_features):
            # Plot original data
            axes[i].plot(X_true[sample_idx, :, i], 'b-', label='True', alpha=0.7)
            
            # Plot imputed data
            axes[i].plot(X_imputed[sample_idx, :, i], 'r--', label='BRITS Imputed', alpha=0.8)
            
            # Highlight missing values
            missing_points = missing_mask[sample_idx, :, i]
            if missing_points.any():
                axes[i].scatter(np.where(missing_points)[0], 
                              X_imputed[sample_idx, missing_points, i], 
                              color='red', s=50, label='Imputed Points', zorder=5)
            
            axes[i].set_title(f'{feature_names[i]} - Sample {sample_idx}')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Usage example and testing
if __name__ == "__main__":
    print("BRITS Implementation Example")
    print("=" * 50)
    
    # Initialize BRITS imputer
    brits_imputer = PyPOTS_BRITS_Imputer(
        n_steps=30,           # 30-day sequences
        n_features=5,         # 5 features as specified
        rnn_hidden_size=256,  # Hidden size for RNN
        epochs=50,            # Number of training epochs
        batch_size=32,        # Batch size
        learning_rate=0.001   # Learning rate
    )
    
    # Example usage (uncomment when you have your CSV file):
    
    # Load and prepare your data
    data, feature_names = brits_imputer.prepare_data(r'E:\FNFLORESR_PROYECTOS\PYPOTS_COMPARISON\data\join_data_31072025.csv')
    
    # Create sequences
    sequences = brits_imputer.create_sequences(data, seq_length=30)
    
    # Split into train and test
    train_size = int(0.8 * len(sequences))
    X_train = sequences[:train_size]
    X_test = sequences[train_size:]
    
    # Introduce missing values in test set for evaluation
    X_test_missing, missing_mask = brits_imputer.introduce_missing_values(X_test, missing_rate=0.1)
    
    # Train BRITS model
    brits_imputer.fit(X_train)
    
    # Perform imputation
    X_imputed = brits_imputer.impute(X_test_missing)
    
    # Evaluate performance
    metrics = brits_imputer.evaluate_performance(X_test, X_imputed, missing_mask)
    
    # Plot results
    brits_imputer.plot_imputation_results(
        X_test, X_imputed, missing_mask, 
        feature_names=feature_names, sample_idx=0
    )
    
    
    print("BRITS imputer initialized successfully!")
    print("Ready for use with your 10-year daily multivariate time series dataset.")


print("BRITS Implementation Created Successfully!")