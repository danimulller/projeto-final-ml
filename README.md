# Project: Loan Default Prediction

This repository analyzes a Kaggle dataset on consumer loans and trains a model to predict the probability that a client will default. The notebooks cover data ingestion, cleaning, exploration and modeling using a random forest classifier.

## Repository structure

```
.
├── data
│   ├── raw
│   │   └── probability_of_default.csv         # original data from Kaggle
│   └── processed
│       └── probability_of_default.csv         # cleaned features
├── notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── models                                      # trained model and helpers
│   ├── default_probability_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
├── requirements.txt
├── test.py                                     # simple prediction script
└── README.md
```

## Dataset

The data comes from the Kaggle dataset **credit-analysis-probability-of-default**. Each record represents a loan application with client demographics and loan characteristics. Key columns after cleaning include:

- `age`
- `income`
- `home_ownership_type` (categorical)
- `employment_length`
- `loan_amount`
- `loan_interest_rate`
- `is_default` (target)
- `loan_percent_income`
- `has_defaulted_before`
- `credit_history_length`

The processed file in `data/processed/` drops unused fields and formats types for modeling.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Kaggle credentials**

   Place your `kaggle.json` file inside a `secrets/` folder. The ingestion notebook uses the Kaggle API to download `probability_of_default.csv`.

3. **Run the notebooks** in order to reproduce the analysis and model training:
   - `01_data_ingestion.ipynb`
   - `02_data_cleaning.ipynb`
   - `03_exploratory_analysis.ipynb`
   - `04_model_training.ipynb`
   - `05_model_evaluation.ipynb`

The trained model and preprocessing objects are saved to the `models/` directory.

## Prediction script

`test.py` demonstrates how to load the trained model and predict the default probability for a single client. Customize the `sample_client` dictionary and run:

```bash
python test.py
```

Example output:

```
Default Probability: 12.34%
```

## Model performance

On the hold-out test set the random forest achieved a ROC AUC of about 0.91.

## Contributing

Pull requests to improve the analysis or documentation are welcome.
