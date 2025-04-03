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
1. Install Dependencies
2. Setup Environment - Create and activate a virtual environment (optional but recommended)
3. 
--------

