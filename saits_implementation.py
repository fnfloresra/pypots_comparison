# Create SAITS implementation


# ===============================================
# SAITS Implementation with PyPOTS
# ===============================================

import numpy as np
import pandas as pd
import torch
from pypots.imputation import SAITS
from pypots.nn.functional import calc_mae, calc_mse, calc_mre
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PyPOTS_SAITS_Imputer:
    def __init__(self, n_steps=30, n_features=5, n_layers=3, d_model=256, 
                 n_heads=8, d_k=32, d_v=32, d_ffn=512, dropout=0.1, epochs=100, 
                 batch_size=32, learning_rate=0.001):
        """
        SAITS (Self-Attention-based Imputation for Time Series) implementation
        
        Args:
            n_steps: Number of time steps in each sequence
            n_features: Number of features (5 for your dataset)
            n_layers: Number of transformer layers
            d_model: Model dimension
            n_heads: Number of attention heads
            d_k: Dimension of key vectors
            d_v: Dimension of value vectors
            d_ffn: Dimension of feed-forward network
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
        """
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_data(self, csv_path):
        """
        Prepare CSV data for PyPOTS SAITS format
        
        Expected CSV format:
        - Date column (will be used as index)
        - 5 feature columns for multivariate time series
        """
        print("Loading and preparing data for SAITS...")
        
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
        print(f"Features: {feature_cols.tolist()}")
        
        return df_filled.values, feature_cols.tolist()
    
    def create_sequences(self, data, seq_length=None, stride=1):
        """
        Create overlapping sequences for SAITS training
        
        Args:
            data: Time series data (time_steps, features)
            seq_length: Length of each sequence (uses self.n_steps if None)
            stride: Step size between sequences
        """
        if seq_length is None:
            seq_length = self.n_steps
            
        sequences = []
        for i in range(0, len(data) - seq_length + 1, stride):
            sequences.append(data[i:i + seq_length])
        
        sequences = np.array(sequences)
        print(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        
        return sequences
    
    def introduce_missing_values(self, data, missing_rate=0.1):
        """
        Introduce missing values for testing (simulating real-world missing data)
        
        Args:
            data: Complete data
            missing_rate: Proportion of values to make missing
        """
        data_with_missing = data.copy()
        
        # Create missing patterns that are more realistic
        # - Random missing
        random_mask = np.random.random(data.shape) < missing_rate * 0.5
        
        # - Block missing (consecutive time steps)
        block_mask = np.zeros_like(data, dtype=bool)
        for sample in range(data.shape[0]):
            for feature in range(data.shape[2]):
                if np.random.random() < missing_rate * 0.3:
                    start_idx = np.random.randint(0, data.shape[1] - 5)
                    block_length = np.random.randint(2, 6)
                    end_idx = min(start_idx + block_length, data.shape[1])
                    block_mask[sample, start_idx:end_idx, feature] = True
        
        # - Feature missing (entire features missing for some time steps)
        feature_mask = np.zeros_like(data, dtype=bool)
        for sample in range(data.shape[0]):
            if np.random.random() < missing_rate * 0.2:
                missing_feature = np.random.randint(0, data.shape[2])
                start_idx = np.random.randint(0, data.shape[1] - 3)
                feature_mask[sample, start_idx:start_idx+3, missing_feature] = True
        
        # Combine all missing patterns
        missing_mask = random_mask | block_mask | feature_mask
        data_with_missing[missing_mask] = np.nan
        
        missing_count = missing_mask.sum()
        missing_percentage = missing_count / data.size * 100
        
        print(f"Introduced {missing_count} missing values ({missing_percentage:.2f}%)")
        print(f"Missing patterns: Random + Block + Feature-wise")
        
        return data_with_missing, missing_mask
    
    def fit(self, X):
        """
        Fit SAITS model
        
        Args:
            X: Training data (n_samples, n_steps, n_features)
        """
        print("Initializing SAITS model...")
        
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
        
        # Initialize SAITS model
        self.model = SAITS(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Training SAITS model...")
        print(f"Model parameters:")
        print(f"  - Layers: {self.n_layers}")
        print(f"  - Model dimension: {self.d_model}")
        print(f"  - Attention heads: {self.n_heads}")
        print(f"  - Dropout: {self.dropout}")
        print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Prepare data in PyPOTS format (dictionary with 'X' key)
        train_data = {"X": X_normalized}
        
        # Train the model
        self.model.fit(train_data)
        self.is_fitted = True
        
        print("SAITS model training completed!")
        return self
    
    def impute(self, X):
        """
        Perform imputation using trained SAITS model
        
        Args:
            X: Data with missing values (n_samples, n_steps, n_features)
            
        Returns:
            Imputed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before imputation!")
        
        print("Performing SAITS imputation...")
        
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
        
        print("SAITS imputation completed!")
        return X_imputed_final
    
    def evaluate_performance(self, X_true, X_imputed, missing_mask):
        """
        Evaluate SAITS imputation performance
        
        Args:
            X_true: Ground truth data
            X_imputed: Imputed data
            missing_mask: Boolean mask indicating missing values
        """
        print("Evaluating SAITS performance...")
        
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
        
        print("SAITS Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def get_attention_maps(self, X, sample_idx=0):
        """
        Extract attention maps from SAITS model for interpretability
        
        Args:
            X: Input data
            sample_idx: Sample index to extract attention for
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")
        
        # This would require access to model internals
        # Implementation depends on PyPOTS version and SAITS model structure
        print("Attention map extraction requires model internal access")
        print("This feature may need PyPOTS version-specific implementation")
        
        return None
    
    def plot_imputation_results(self, X_true, X_imputed, missing_mask, 
                               feature_names=None, sample_idx=0):
        """
        Plot imputation results for visualization with attention to transformer patterns
        
        Args:
            X_true: Ground truth data
            X_imputed: Imputed data
            missing_mask: Missing value mask
            feature_names: Names of features
            sample_idx: Sample index to plot
        """
        if feature_names is None:
            feature_names = [f'Feature_{i+1}' for i in range(self.n_features)]
        
        fig, axes = plt.subplots(self.n_features, 1, figsize=(15, 3*self.n_features))
        if self.n_features == 1:
            axes = [axes]
        
        for i in range(self.n_features):
            # Plot original data
            axes[i].plot(X_true[sample_idx, :, i], 'b-', label='Ground Truth', 
                        linewidth=2, alpha=0.8)
            
            # Plot imputed data
            axes[i].plot(X_imputed[sample_idx, :, i], 'r--', label='SAITS Imputed', 
                        linewidth=2, alpha=0.9)
            
            # Highlight missing values with different colors for different patterns
            missing_points = missing_mask[sample_idx, :, i]
            if missing_points.any():
                # Single missing points
                single_missing = []
                block_start = []
                
                missing_indices = np.where(missing_points)[0]
                for idx in missing_indices:
                    # Check if it's part of a block (consecutive missing values)
                    is_block = False
                    if idx > 0 and missing_points[idx-1]:
                        is_block = True
                    if idx < len(missing_points)-1 and missing_points[idx+1]:
                        is_block = True
                    
                    if is_block:
                        block_start.append(idx)
                    else:
                        single_missing.append(idx)
                
                # Plot single missing points
                if single_missing:
                    axes[i].scatter(single_missing, 
                                  X_imputed[sample_idx, single_missing, i], 
                                  color='red', s=60, label='Single Missing Imputed', 
                                  zorder=5, marker='o')
                
                # Plot block missing points
                if block_start:
                    axes[i].scatter(block_start, 
                                  X_imputed[sample_idx, block_start, i], 
                                  color='orange', s=60, label='Block Missing Imputed', 
                                  zorder=5, marker='s')
            
            axes[i].set_title(f'{feature_names[i]} - SAITS Imputation (Sample {sample_idx})', 
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Time Steps', fontsize=10)
            axes[i].set_ylabel('Normalized Value', fontsize=10)
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3)
            
            # Add shaded regions for missing blocks
            in_block = False
            block_start_idx = 0
            for j in range(len(missing_points)):
                if missing_points[j] and not in_block:
                    block_start_idx = j
                    in_block = True
                elif not missing_points[j] and in_block:
                    axes[i].axvspan(block_start_idx, j-1, alpha=0.2, color='yellow', 
                                   label='Missing Block' if i == 0 else '')
                    in_block = False
            
            # Handle case where block extends to end
            if in_block:
                axes[i].axvspan(block_start_idx, len(missing_points)-1, 
                               alpha=0.2, color='yellow')
        
        plt.tight_layout()
        plt.suptitle('SAITS (Self-Attention) Imputation Results', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.show()
    
    def compare_with_baseline(self, X_true, X_imputed, missing_mask, 
                             baseline_method='mean'):
        """
        Compare SAITS performance with simple baseline methods
        
        Args:
            X_true: Ground truth data
            X_imputed: SAITS imputed data
            missing_mask: Missing value mask
            baseline_method: 'mean', 'median', 'forward_fill'
        """
        print(f"Comparing SAITS with {baseline_method} baseline...")
        
        # Create baseline imputation
        X_baseline = X_true.copy()
        
        if baseline_method == 'mean':
            for i in range(self.n_features):
                feature_mean = np.nanmean(X_true[:, :, i])
                X_baseline[missing_mask[:, :, i], i] = feature_mean
        elif baseline_method == 'median':
            for i in range(self.n_features):
                feature_median = np.nanmedian(X_true[:, :, i])
                X_baseline[missing_mask[:, :, i], i] = feature_median
        elif baseline_method == 'forward_fill':
            # Forward fill within each sample
            for sample in range(X_baseline.shape[0]):
                df_sample = pd.DataFrame(X_baseline[sample])
                df_filled = df_sample.ffill().bfill()
                X_baseline[sample] = df_filled.values
        
        # Calculate baseline metrics
        baseline_mae = calc_mae(X_baseline[missing_mask], X_true[missing_mask])
        baseline_mse = calc_mse(X_baseline[missing_mask], X_true[missing_mask])
        baseline_mre = calc_mre(X_baseline[missing_mask], X_true[missing_mask])
        
        # Calculate SAITS metrics
        saits_mae = calc_mae(X_imputed[missing_mask], X_true[missing_mask])
        saits_mse = calc_mse(X_imputed[missing_mask], X_true[missing_mask])
        saits_mre = calc_mre(X_imputed[missing_mask], X_true[missing_mask])
        
        # Create comparison
        comparison = {
            'Baseline': {
                'Method': baseline_method,
                'MAE': float(baseline_mae),
                'MSE': float(baseline_mse),
                'MRE': float(baseline_mre),
                'RMSE': float(np.sqrt(baseline_mse))
            },
            'SAITS': {
                'MAE': float(saits_mae),
                'MSE': float(saits_mse),
                'MRE': float(saits_mre),
                'RMSE': float(np.sqrt(saits_mse))
            },
            'Improvement': {
                'MAE': f"{((baseline_mae - saits_mae) / baseline_mae * 100):.2f}%",
                'MSE': f"{((baseline_mse - saits_mse) / baseline_mse * 100):.2f}%",
                'MRE': f"{((baseline_mre - saits_mre) / baseline_mre * 100):.2f}%",
                'RMSE': f"{((np.sqrt(baseline_mse) - np.sqrt(saits_mse)) / np.sqrt(baseline_mse) * 100):.2f}%"
            }
        }
        
        print("\\nComparison Results:")
        print(f"{'Metric':<8} {'Baseline':<12} {'SAITS':<12} {'Improvement':<12}")
        print("-" * 50)
        for metric in ['MAE', 'MSE', 'MRE', 'RMSE']:
            print(f"{metric:<8} {comparison['Baseline'][metric]:<12.6f} "
                  f"{comparison['SAITS'][metric]:<12.6f} {comparison['Improvement'][metric]:<12}")
        
        return comparison

# Usage example and testing
if __name__ == "__main__":
    print("SAITS Implementation Example")
    print("=" * 50)
    
    # Initialize SAITS imputer with transformer-specific parameters
    saits_imputer = PyPOTS_SAITS_Imputer(
        n_steps=30,           # 30-day sequences
        n_features=5,         # 5 features as specified
        n_layers=3,           # Number of transformer layers
        d_model=256,          # Model dimension
        n_heads=8,            # Number of attention heads
        d_k=32,               # Key dimension
        d_v=32,               # Value dimension
        d_ffn=512,            # Feed-forward network dimension
        dropout=0.1,          # Dropout rate
        epochs=50,            # Number of training epochs
        batch_size=32,        # Batch size
        learning_rate=0.001   # Learning rate
    )
    
    # Example usage (uncomment when you have your CSV file):
    
    # Load and prepare your data
    data, feature_names = saits_imputer.prepare_data(r'E:\FNFLORESR_PROYECTOS\PYPOTS_COMPARISON\data\join_data_31072025.csv')

    # Create sequences with overlapping windows
    sequences = saits_imputer.create_sequences(data, seq_length=30, stride=7)
    
    # Split into train and test
    train_size = int(0.8 * len(sequences))
    X_train = sequences[:train_size]
    X_test = sequences[train_size:]
    
    # Introduce realistic missing patterns in test set
    X_test_missing, missing_mask = saits_imputer.introduce_missing_values(X_test, missing_rate=0.15)
    
    # Train SAITS model
    saits_imputer.fit(X_train)
    
    # Perform imputation
    X_imputed = saits_imputer.impute(X_test_missing)
    
    # Evaluate performance
    metrics = saits_imputer.evaluate_performance(X_test, X_imputed, missing_mask)
    
    # Compare with baseline
    comparison = saits_imputer.compare_with_baseline(X_test, X_imputed, missing_mask, 'mean')
    
    # Plot results
    saits_imputer.plot_imputation_results(
        X_test, X_imputed, missing_mask, 
        feature_names=feature_names, sample_idx=0
    )
    
    
    print("SAITS imputer initialized successfully!")
    print("Ready for use with your 10-year daily multivariate time series dataset.")
    print("This transformer-based model excels at capturing complex temporal patterns.")


print("SAITS Implementation Created Successfully!")