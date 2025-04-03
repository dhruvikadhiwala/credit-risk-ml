# Enhanced Credit Risk Assessment

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Machine Learning for credit  default prediction with Lending Club Data

This project uses supervised and unsupervised learning techniques (Logistic Regression, Random Forest, Isolation Forest) to predict credit default risk. It follows the Cookiecutter Data Science structure to promote reproducibility, modularity, and scalability.

## Project Organization

```
├── LICENSE – Defines the legal terms for project reuse and distribution.
├── Makefile - Contains convenient CLI commands for automation (e.g., make train, make data).
├── README.md – Describes the project, its structure, and usage instructions.
├── requirements.txt – Lists all Python dependencies required to run the project.
├── pyproject.toml – Configuration files for code formatting, linting, and tool integrations.
├── setup.cfg – Configuration files for code formatting, linting, and tool integrations.
├── data/
│   ├── raw/ – Stores the untouched dataset (cleaned_data.csv) as originally provided.
│   │   └── cleaned_data.csv
│   ├── processed/ – Holds the final transformed dataset (normalized_data.csv) ready for modeling.
│   │   └── normalized_data.csv
├── notebooks/ 
│   └── eda.ipynb – Jupyter notebook used for exploratory data analysis, including visualizations and summary statistics.
├── reports/
│   ├── model_outputs.csv – Probability outputs for each model on the dataset.
│   ├── model_comparison.csv – Summary of performance metrics across all models.
│   ├── hybrid_model_results.csv – Final results using a hybrid anomaly-based approach.
│   └── figures/ – Contains visualizations such as feature_importance.png and threshold_impact.png used for analysis and reporting.
│       ├── feature_importance.png
│       └── threshold_impact.png
├── credit_risk/
│   ├── __init__.py – Marks this directory as a Python package.
│   ├── config.py – Stores global paths and project-level configurations.
│   ├── dataset.py  – Loads the raw data, applies feature engineering (e.g., scaling, encoding), and outputs processed data.
│   ├── features.py – Contains helper functions for transforming or selecting features.
│   ├── plots.py  – Defines reusable functions for generating visualizations and plots.
│   └── modeling/
│       ├── __init__.py
│       ├── train.py – Main script for training all models and saving results/figures.
│       └── predict.py – Placeholder for inference logic on unseen data (e.g., future loan applications).

```
## Setup Instructions
1. Install Dependencies - Run pip install -r requirements.txt
2. Setup Environment - Create and activate a virtual environment (optional but recommended)
3. Train & Evaluate Models - Run python credit_risk/modeling/train.py

## How To Use
1. Data Processing - To load and normalize the data run python credit_risk/dataset.py
2. Run EDA - Navigate to notebooks/eda.ipynb to explore distributions, correlations, and class balance.
3. Train Models - Run python -m credit_risk.modeling.train. This will train all models on normalized_data.csv. This save outputs to reports/model_outputs.csv, reports/figures/feature_importance.png, reports/model_comparison.csv, and reports/hybrid_model_results.csv.

## Key Features
1. Logistic Regression: Simple & interpretable
2. Random Forest: Captures nonlinear patterns
3. Isolation Forest: Unsupervised anomaly detection
4. Hybrid Approach: Combines unsupervised & supervised methods
5. AUPRC, Ranking, FPR: Comprehensive evaluation metrics
6. Threshold Analysis: Understand tradeoffs in risk classification

## Reproducibility 
Every script is designed to:
1. Use consistent input/output directories
2. Be reusable & independent
3. Save logs for traceability

## Dependencies 
1. Python 3.9+
2. pandas
3. scikit-learn
4. seaborn
5. matplotlib
6. loguru
7. imbalanced-learn
8. typer
--------

