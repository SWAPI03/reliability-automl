"""Generate large synthetic datasets for comprehensive testing."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_large_dataset(n_samples=10000, n_features=20, noise_level=0.15, 
                          missing_pct=0.08, duplicate_pct=0.05, 
                          outlier_pct=0.03, filename='large_dataset_10000.csv'):
    """
    Generate a large synthetic dataset with controlled noise characteristics.
    
    Parameters:
    - n_samples: Number of rows
    - n_features: Number of features
    - noise_level: Percentage of label noise (mislabeling)
    - missing_pct: Percentage of missing values
    - duplicate_pct: Percentage of duplicate rows
    - outlier_pct: Percentage of outlier values
    """
    print(f"Generating dataset: {n_samples} rows, {n_features} features...")
    
    # Generate base classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_repeated=0,
        n_classes=2,
        flip_y=noise_level,  # Label noise
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add missing values
    n_missing = int(n_samples * n_features * missing_pct)
    for _ in range(n_missing):
        row_idx = np.random.randint(0, n_samples)
        col_idx = np.random.randint(0, n_features)
        df.iloc[row_idx, col_idx] = np.nan
    
    # Add duplicates
    n_duplicates = int(n_samples * duplicate_pct)
    if n_duplicates > 0:
        duplicate_indices = np.random.choice(n_samples, n_duplicates, replace=True)
        duplicate_rows = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Add outliers
    n_outliers = int(n_samples * outlier_pct)
    for _ in range(n_outliers):
        row_idx = np.random.randint(0, len(df))
        col_idx = np.random.randint(0, n_features)
        # Set to extreme value (5 standard deviations away)
        mean_val = df.iloc[:, col_idx].mean()
        std_val = df.iloc[:, col_idx].std()
        df.iloc[row_idx, col_idx] = mean_val + (5 * std_val * np.random.choice([-1, 1]))
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(filename, index=False)
    print(f"✓ Saved {filename}")
    print(f"  Final shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.2f}%)")
    print(f"  Duplicates: {df.duplicated().sum()}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    # Generate multiple large datasets with varying characteristics
    
    print("\n=== Dataset 1: 10,000 rows, moderate noise ===")
    generate_large_dataset(
        n_samples=10000,
        n_features=20,
        noise_level=0.15,
        missing_pct=0.08,
        duplicate_pct=0.05,
        outlier_pct=0.03,
        filename='large_dataset_10000.csv'
    )
    
    print("\n=== Dataset 2: 15,000 rows, high noise ===")
    generate_large_dataset(
        n_samples=15000,
        n_features=25,
        noise_level=0.25,
        missing_pct=0.12,
        duplicate_pct=0.08,
        outlier_pct=0.05,
        filename='large_dataset_15000_noisy.csv'
    )
    
    print("\n=== Dataset 3: 20,000 rows, low noise (clean) ===")
    generate_large_dataset(
        n_samples=20000,
        n_features=15,
        noise_level=0.05,
        missing_pct=0.03,
        duplicate_pct=0.02,
        outlier_pct=0.01,
        filename='large_dataset_20000_clean.csv'
    )
    
    print("\n=== Dataset 4: 12,000 rows, extreme noise ===")
    generate_large_dataset(
        n_samples=12000,
        n_features=30,
        noise_level=0.35,
        missing_pct=0.18,
        duplicate_pct=0.12,
        outlier_pct=0.08,
        filename='large_dataset_12000_extreme.csv'
    )
    
    print("\n✓ All datasets generated successfully!")
