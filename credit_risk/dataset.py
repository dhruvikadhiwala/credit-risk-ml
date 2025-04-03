from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from credit_risk.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "cleaned_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "normalized_data.csv",
):
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    # Create target column from encoded data
    if "loan_status_Default" in df.columns:
        df["default"] = df["loan_status_Default"]
        logger.info("'default' column created from 'loan_status_Default'")
    else:
        logger.error("'loan_status_Default' not found.")
        return

    # Drop the original loan_status one-hot columns if you want
    loan_status_cols = [col for col in df.columns if col.startswith("loan_status_")]
    df.drop(columns=loan_status_cols, inplace=True)

    logger.info("Scaling numeric features (excluding 'default')...")
    feature_cols = df.drop(columns=["default"]).columns
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv(output_path, index=False)
    logger.success(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    app()