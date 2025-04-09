import pandas as pd 
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the mall customer dataset"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, save_cleaned_path=None):
    """Clean and prepare data for clustering"""
    # Drop CustomerID column if it exists
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Encode Gender column if it exists
    if 'Gender' in df.columns:
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # Handle missing values (fill with mean for numeric, mode for categorical)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)

    # Save cleaned data (before scaling) if path is provided
    if save_cleaned_path:
        df.to_csv(save_cleaned_path, index=False)
        print(f"✅ Cleaned data saved to: {save_cleaned_path}")

    # Scale features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    return scaled_features

# Example usage
if __name__ == "__main__":
    raw_data_path = "data/mall_customers.csv"
    cleaned_data_path = "data/cleaned_mall_customers.csv"
    
    df = load_data(raw_data_path)
    preprocess_data(df, save_cleaned_path=cleaned_data_path)
