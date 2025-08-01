# Test the comparison framework with actual data

import sys
import os
sys.path.append('src')

from src.comparison_framework import PyPOTS_Comparison_Framework

# Initialize comparison framework
comparison = PyPOTS_Comparison_Framework(
    data_path=r'E:\FNFLORESR_PROYECTOS\PYPOTS_COMPARISON\data\join_data_31072025.csv',
    sequence_length=30,
    missing_rates=[0.1, 0.15]  # Reduced missing rates for faster testing
)

print("Testing data loading...")
try:
    sequences, feature_names = comparison.load_and_prepare_data()
    print(f"✓ Data loaded successfully: {sequences.shape}")
    print(f"✓ Features: {feature_names}")
except Exception as e:
    print(f"✗ Error loading data: {e}")

print("\nTesting model initialization...")
try:
    comparison.initialize_models()
    print(f"✓ Models initialized: {list(comparison.models.keys())}")
except Exception as e:
    print(f"✗ Error initializing models: {e}")

print("\nComparison framework test completed!")
