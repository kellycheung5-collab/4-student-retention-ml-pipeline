import os

import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads raw CSV data with semicolon delimiter and cleans column names."""
    df = pd.read_csv(filepath, sep=";")
    df.columns = df.columns.str.strip()
    return df


def process_datasets(df: pd.DataFrame):
    """Generates the full-information and early-warning datasets with binary targets."""
    # Create binary target (Dropout = 1, Graduate/Enrolled = 0)
    df["target_binary"] = df["Target"].apply(
        lambda x: 1 if x == "Dropout" else 0
    )

    # Full information dataset
    df_full = df.copy()

    # Early warning dataset (drop 2nd semester features to prevent leakage)
    second_sem_cols = [col for col in df.columns if "2nd sem" in col.lower()]
    df_early = df.drop(columns=second_sem_cols)

    return df_full, df_early


def save_processed_data(
    df_full: pd.DataFrame, df_early: pd.DataFrame, output_dir: str
):
    """Saves processed dataframes to CSV files in data/processed/."""
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "student_data_full.csv")
    early_path = os.path.join(output_dir, "student_data_early_warning.csv")

    df_full.to_csv(full_path, index=False)
    df_early.to_csv(early_path, index=False)

    print(f"Saved full-information dataset to: {full_path}")
    print(f"Saved early-warning dataset to:    {early_path}")


if __name__ == "__main__":
    raw_csv_path = "data/raw/data.csv"
    processed_dir = "data/processed"

    df_raw = load_raw_data(raw_csv_path)
    df_full, df_early = process_datasets(df_raw)
    save_processed_data(df_full, df_early, processed_dir)