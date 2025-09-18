# src/preprocessing.py
import json
import pandas as pd
import os
from config import DATA_PATH_RAW, DATA_PATH_PROCESSED

def preprocess_json_to_csv():
    os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)
    raw_file = os.path.join(DATA_PATH_RAW, "technology_raw.json")
    processed_file = os.path.join(DATA_PATH_PROCESSED, "technology_raw.csv")

    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

if __name__ == "__main__":
    preprocess_json_to_csv()
