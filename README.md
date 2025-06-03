# Project: Cancer Treatment Cost Analysis

This repository explores a public Kaggle dataset with information about cancer patients worldwide (2015–2024). The notebooks guide you from data acquisition through exploratory analysis to modeling.

## Repository structure

```
.
├── data
│   ├── raw
│   │   └── kaggle_cancer_patients_raw.csv      # original data from Kaggle
│   └── processed
│       └── kaggle_cancer_patients_processed.csv  # cleaned data
├── notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_model_export.ipynb
├── requirements.txt
└── README.md                                    # this file
```

## Dataset

The raw dataset (`kaggle_cancer_patients_raw.csv`) includes:
- Patient demographics (age, gender, country, year)
- Risk factors (genetic risk, air pollution, alcohol use, smoking, obesity level)
- Cancer information (type, stage)
- Treatment cost in USD and survival years
- Target severity score

After cleaning, the processed dataset removes identifiers and retains the key features needed for analysis.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The notebooks use additional packages such as `matplotlib`, `seaborn`, and `scikit-learn`. Install them as needed:

   ```bash
   pip install matplotlib seaborn scikit-learn lifelines
   ```

2. **Kaggle credentials**

   To download the dataset with the Kaggle API, place your `kaggle.json` file under a folder called `secrets/` (ignored by Git). The `01_data_ingestion.ipynb` notebook uses the Kaggle API to fetch and unpack the dataset.

3. **Running the notebooks**

   The notebooks are numbered in order:
   - `01_data_ingestion.ipynb` – Download data from Kaggle.
   - `02_data_cleaning.ipynb` – Basic cleaning and feature selection.
   - `03_exploratory_analysis.ipynb` – Visualization and statistical exploration.
   - `04_feature_engineering.ipynb` – Placeholder for engineered features.
   - `05_model_training.ipynb` – Placeholder for training models.
   - `06_model_evaluation.ipynb` – Placeholder for evaluation metrics.
   - `07_model_export.ipynb` – Placeholder for exporting the trained model.

   Execute each notebook sequentially to reproduce the workflow.

## Project status

Data ingestion, cleaning, and some exploratory analysis are implemented. The notebooks for feature engineering, model training, evaluation, and export currently contain minimal code and can be extended to build predictive models.

## Contributing

Feel free to fork the repository and open pull requests. Improvements to analysis, modeling, or documentation are welcome!

