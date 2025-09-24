"""
Example usage of intelligent data cleaning with OpenAI agents
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Add parent directory to path (automl_platform is already in the path)
# No need to modify sys.path if running from project root
# If running this file directly, uncomment the next line:
# sys.path.append(str(Path(__file__).parent.parent.parent))

from ..agents import DataCleaningOrchestrator, AgentConfig
from ..data_prep import EnhancedDataPreprocessor
from ..config import AutoMLConfig


async def example_intelligent_cleaning():
    """
    Demonstrate intelligent data cleaning with OpenAI agents
    """
    
    # 1. Generate sample dataset (Finance sector example)
    print("📊 Generating sample financial dataset...")
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'transaction_id': range(1, n_samples + 1),
        'amount': np.random.lognormal(5, 2, n_samples),
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'customer_id': np.random.choice(['C' + str(i) for i in range(100)], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', None], n_samples),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'PayPal'], n_samples),
        'currency': np.random.choice(['EUR', 'USD', 'GBP', '€', '$'], n_samples),  # Inconsistent formats
        'status': np.random.choice(['Completed', 'completed', 'COMPLETED', 'Pending', None], n_samples),
        'risk_score': np.random.uniform(0, 100, n_samples)
    })
    
    # Add some data quality issues
    df.loc[np.random.choice(df.index, 50), 'amount'] = np.nan  # Missing values
    df.loc[np.random.choice(df.index, 30), 'amount'] *= 100  # Outliers
    df.loc[np.random.choice(df.index, 20), 'customer_id'] = None  # Missing customer IDs
    
    # Add duplicates
    duplicate_rows = df.sample(20)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    print(f"✅ Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"⚠️ Data issues introduced: missing values, outliers, duplicates, inconsistent formats")
    
    # 2. Configure intelligent cleaning
    print("\n🤖 Configuring OpenAI agents...")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    # User context for cleaning
    user_context = {
        "secteur_activite": "finance",
        "target_variable": "risk_score",
        "contexte_metier": "Transaction risk prediction for fraud detection",
        "language": "en"
    }
    
    # Create agent configuration
    agent_config = AgentConfig(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4-1106-preview",
        enable_web_search=True,
        enable_file_operations=True,
        max_iterations=3,
        timeout_seconds=300,
        max_cost_per_dataset=5.00,
        user_context=user_context
    )
    
    # 3. Run intelligent cleaning
    print("\n🔄 Starting intelligent data cleaning process...")
    print(f"   Sector: {user_context['secteur_activite']}")
    print(f"   Target: {user_context['target_variable']}")
    
    orchestrator = DataCleaningOrchestrator(agent_config)
    
    # Display initial data quality
    print("\n📈 Initial Data Quality:")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicates: {df.duplicated().sum()}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Clean the dataset
    cleaned_df, report = await orchestrator.clean_dataset(df, user_context)
    
    # 4. Display results
    print("\n✨ Cleaning Results:")
    print(f"   Original shape: {df.shape}")
    print(f"   Cleaned shape: {cleaned_df.shape}")
    print(f"   Quality score: {report.get('quality_metrics', {}).get('quality_score', 0):.1f}/100")
    
    print("\n📊 Transformations Applied:")
    for i, trans in enumerate(report.get('transformations', [])[:5], 1):
        print(f"   {i}. {trans.get('action')} on column '{trans.get('column', 'N/A')}'")
        if trans.get('params'):
            print(f"      Parameters: {trans['params']}")
    
    if len(report.get('transformations', [])) > 5:
        print(f"   ... and {len(report['transformations']) - 5} more transformations")
    
    print("\n🔍 Validation Sources Referenced:")
    for source in report.get('validation_sources', [])[:3]:
        print(f"   - {source}")
    
    print("\n📈 Final Data Quality:")
    print(f"   Missing values: {cleaned_df.isnull().sum().sum()}")
    print(f"   Duplicates: {cleaned_df.duplicated().sum()}")
    print(f"   Completeness: {(1 - cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1])) * 100:.1f}%")
    
    # 5. Save results
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    cleaned_df.to_csv(output_dir / "cleaned_data.csv", index=False)
    print(f"\n💾 Cleaned data saved to: {output_dir / 'cleaned_data.csv'}")
    
    # Save cleaning report
    import json
    with open(output_dir / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Cleaning report saved to: {output_dir / 'cleaning_report.json'}")
    
    return cleaned_df, report


async def example_with_automl_integration():
    """
    Example using intelligent cleaning integrated with AutoML platform
    """
    print("\n" + "="*60)
    print("🚀 Example with AutoML Platform Integration")
    print("="*60)
    
    # Create AutoML configuration
    config = AutoMLConfig()
    config.expert_mode = False  # Use simplified mode
    config.llm.enabled = True
    config.llm.enable_data_cleaning = True
    
    # Create sample dataset
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C', None], 100),
        'feature3': np.random.uniform(0, 100, 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 10), 'feature1'] = np.nan
    
    # Create preprocessor with intelligent cleaning
    prep_config = {
        'enable_intelligent_cleaning': True,
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'openai_cleaning_model': 'gpt-4-1106-preview',
        'max_cleaning_cost_per_dataset': 5.00,
        'enable_web_search': True
    }
    
    preprocessor = EnhancedDataPreprocessor(prep_config)
    
    # User context
    user_context = {
        "secteur_activite": "general",
        "target_variable": "target",
        "contexte_metier": "Binary classification task"
    }
    
    # Run intelligent cleaning
    print("\n🔄 Running intelligent cleaning through AutoML integration...")
    cleaned_df = await preprocessor.intelligent_clean(df, user_context)
    
    print(f"\n✅ Cleaning completed!")
    print(f"   Original shape: {df.shape}")
    print(f"   Cleaned shape: {cleaned_df.shape}")
    
    if hasattr(preprocessor, 'cleaning_report'):
        print(f"   Quality score: {preprocessor.cleaning_report.get('quality_metrics', {}).get('quality_score', 0):.1f}/100")


async def example_sector_specific():
    """
    Examples for different business sectors
    """
    print("\n" + "="*60)
    print("🏢 Sector-Specific Cleaning Examples")
    print("="*60)
    
    # Healthcare sector example
    print("\n🏥 Healthcare Sector Example:")
    
    healthcare_df = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'diagnosis_code': ['A01.1', 'B02.2', 'Invalid', 'C03.3', None],
        'admission_date': ['2023-01-01', '2023-01-02', '01/03/2023', '2023-01-04', '2023-01-05'],
        'lab_result': [120, 145, 999999, 110, 135],  # Include outlier
        'medication': ['Aspirin', 'aspirin', 'ASPIRIN', 'Ibuprofen', None]
    })
    
    healthcare_context = {
        "secteur_activite": "sante",
        "target_variable": "lab_result",
        "contexte_metier": "Patient lab result analysis for treatment optimization"
    }
    
    print(f"   Sample healthcare data with {len(healthcare_df)} records")
    print("   Issues: inconsistent formats, invalid codes, outliers")
    
    # Retail sector example
    print("\n🛍️ Retail Sector Example:")
    
    retail_df = pd.DataFrame({
        'sku': ['SKU001', 'SKU002', 'SKU-003', '004', 'SKU005'],
        'product_name': ['Laptop', 'laptop', 'Mouse', 'Keyboard', None],
        'price': [999.99, 1099.99, -25.50, 45.00, 29.99],  # Negative price
        'quantity': [10, 15, 5, 0, -2],  # Negative quantity
        'category': ['Electronics', 'electronic', 'ELECTRONICS', 'Electronics', 'Accessories']
    })
    
    retail_context = {
        "secteur_activite": "retail",
        "target_variable": "quantity",
        "contexte_metier": "Inventory management and demand forecasting"
    }
    
    print(f"   Sample retail data with {len(retail_df)} records")
    print("   Issues: inconsistent SKU format, negative values, category variations")
    
    print("\n💡 These examples demonstrate sector-specific validation and cleaning")


def main():
    """
    Main function to run examples
    """
    print("🎯 AutoML Platform - Intelligent Data Cleaning Examples")
    print("="*60)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Warning: OPENAI_API_KEY not set")
        print("To run these examples with actual OpenAI agents:")
        print("1. Get your API key from https://platform.openai.com")
        print("2. Set the environment variable:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("\nThe examples will show the structure but won't make actual API calls.\n")
    
    # Run examples
    loop = asyncio.get_event_loop()
    
    try:
        # Run main example
        print("\n1️⃣ Running main intelligent cleaning example...")
        loop.run_until_complete(example_intelligent_cleaning())
        
        # Run AutoML integration example
        print("\n2️⃣ Running AutoML integration example...")
        loop.run_until_complete(example_with_automl_integration())
        
        # Show sector-specific examples
        print("\n3️⃣ Showing sector-specific examples...")
        loop.run_until_complete(example_sector_specific())
        
        print("\n✅ All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
