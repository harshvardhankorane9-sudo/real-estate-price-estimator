import pandas as pd

def generate_sample_data():
    df = pd.read_csv("data/Bengaluru_House_Data.csv")
    df_sample = df.head(500)  # Take first 500 rows as sample
    df_sample.to_csv("data/Bengaluru_House_Data_sample.csv", index=False)
    print("Sample dataset created at data/Bengaluru_House_Data_sample.csv")

if __name__ == "__main__":
    generate_sample_data()
