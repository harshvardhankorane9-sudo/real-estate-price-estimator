import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data with shape: {df.shape}")
    return df

def basic_info(df: pd.DataFrame):
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())

def visualize_distribution(df: pd.DataFrame):
    for col in ['price', 'total_sqft', 'bath']:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=40, kde=True, color='teal')
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

def correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = Path("../data/Bengaluru_House_Data.csv")
    df = load_data(data_path)
    basic_info(df)
    visualize_distribution(df)
    correlation_heatmap(df)
