# Quick demo of the comparison framework

import sys
import os
sys.path.append('src')

from src.comparison_framework import PyPOTS_Comparison_Framework

print("=" * 60)
print("PyPOTS Methods Comparison Demo")
print("=" * 60)

# Initialize comparison framework with reduced complexity for demo
comparison = PyPOTS_Comparison_Framework(
    data_path=r'E:\FNFLORESR_PROYECTOS\PYPOTS_COMPARISON\data\join_data_31072025.csv',
    sequence_length=15,  # Shorter sequences for faster demo
    missing_rates=[0.1]   # Single missing rate for demo
)

print("\n🔄 Running quick comparison demo...")
print("   (This may take a few minutes as models train)")

try:
    # Run the comparison
    results = comparison.run_comprehensive_comparison()
    
    print("\n" + "=" * 40)
    print("📊 RESULTS SUMMARY")
    print("=" * 40)
    
    for model_name, model_results in results['model_comparison'].items():
        print(f"\n{model_name}:")
        print(f"  Training time: {model_results.get('training_time', 0):.2f}s")
        
        missing_key = 'missing_0.1'
        if missing_key in model_results and 'metrics' in model_results[missing_key]:
            metrics = model_results[missing_key]['metrics']
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  Imputation time: {model_results[missing_key].get('imputation_time', 0):.2f}s")
        else:
            print("  Error occurred during testing")
    
    print("\n🎨 Generating visualizations...")
    comparison.create_comparison_visualizations()
    
    print("\n📄 Generating reports...")
    comparison.generate_report()
    
    print("\n✅ Demo completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during comparison: {e}")
    import traceback
    traceback.print_exc()
