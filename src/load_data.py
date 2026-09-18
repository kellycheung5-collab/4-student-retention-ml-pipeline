from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def load_and_preprocess_data(
    input_filename: str = "data.csv",
):
    possible_paths = [
        BASE_DIR / "data" / "raw" / input_filename,
        BASE_DIR / "data" / "raw" / "student_data_early_warning.csv",
        BASE_DIR / input_filename,
        Path(input_filename),
    ]

    input_path = next((p for p in possible_paths if p.exists()), None)

    if input_path is None:
        raise FileNotFoundError(
            f"Could not locate '{input_filename}'. Ensured paths searched: {possible_paths}"
        )

    # Read CSV with separator auto-detection (handles both ',' and ';')
    df = pd.read_csv(input_path, sep=None, engine="python")

    # Clean whitespace or tabs from column names (e.g., 'Daytime/evening attendance\t')
    df.columns = df.columns.str.strip()

    # Map target column to binary dropout_risk
    if "dropout_risk" in df.columns:
        pass
    elif "target_binary" in df.columns:
        df["dropout_risk"] = df["target_binary"]
    elif "Target" in df.columns:
        df["dropout_risk"] = (df["Target"] == "Dropout").astype(int)
    else:
        raise KeyError(
            f"Target column missing from raw CSV dataset. Available columns: {list(df.columns)}"
        )

    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "student_data_early_warning.csv", index=False)
    df.to_csv(output_dir / "student_data_full.csv", index=False)
    print(f"Processed datasets created successfully at {output_dir}")


if __name__ == "__main__":
    load_and_preprocess_data()