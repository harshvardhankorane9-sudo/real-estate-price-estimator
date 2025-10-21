from pathlib import Path
import pandas as pd
from typing import Optional, Union

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def default_raw_csv_path() -> Path:
    return project_root() / "data" / "Bengaluru_House_Data.csv"

def load_raw_data(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else default_raw_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(df.shape)
    print(df.head())
