# Freight Rate Prediction - ML Assessment

This repository contains my solution for the **Spotter Freight Rate Prediction Challenge**.

The objective is to build a machine-learning model that predicts freight load rates using historical shipment information and then generate predictions for both the supplied validation dataset and the fixed December 2025 scenario.

## Project Structure

```text
spotter-freight-rate-ml-assessment/
│
├── notebook/
│   └── freight_rate_prediction.ipynb
│
├── outputs/
│   ├── validation_predictions.csv
│   └── candidate_december.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

The raw assessment datasets are not included in this repository.

## Approach

The workflow consists of:

1. Data inspection and quality checks
2. Missing-value handling
3. Invalid-value cleaning
4. Date-based feature engineering
5. Chronological train/validation split
6. LightGBM regression model training
7. Local validation
8. Retraining on the complete labeled dataset
9. Validation-set prediction
10. December 2025 rate prediction
11. Output validation using the supplied scorer

## Data Preprocessing

Several preprocessing steps were applied before training.

### Weight Cleaning

Shipment weight contained missing and invalid negative values.

Negative values were converted to their absolute values:

```python
weight_clean = abs(weight)
```

Missing weights were imputed using the median weight for the corresponding equipment type.

If an equipment-specific median was unavailable, the overall training-set median was used.

### Date Features

The shipment date was transformed into features that allow the model to capture temporal patterns.

Features include:

- day of week
- month
- day of year
- cyclic month representation
- cyclic day-of-year representation

Cyclic transformations were used because calendar variables are periodic.

### Geographic Features

The model uses geographic information including:

- pickup latitude
- pickup longitude
- delivery latitude
- delivery longitude
- latitude difference
- longitude difference
- shipment distance

These features allow the model to learn geographic effects without relying solely on city names.

### Equipment

`equipment` is treated as a categorical feature by LightGBM.

### Load ID

`load_id` is only an identifier and is therefore excluded from model training.

## Validation Strategy

A **chronological validation split** was used rather than a random train/test split.

The development dataset covers freight activity over time, while the final validation dataset represents later dates.

Therefore, the local split was designed to simulate prediction on future shipments:

```text
Training period:
January 2025 - August 2025

Validation period:
September 2025 - October 2025
```

This reduces temporal leakage and provides a more realistic estimate of future-model performance.

After model selection and validation, the model was retrained using the complete labeled training dataset.

## Model

The final model is a **LightGBM Regressor**.

Main configuration:

```python
LGBMRegressor(
    objective="regression_l1",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    reg_lambda=10.0,
    random_state=42
)
```

The L1 regression objective was selected because freight-rate prediction can contain outliers, and absolute-error-based optimization is less sensitive to extreme observations than squared-error optimization.

## Features Used

The main features are:

```text
distance
weight_clean
pickup_lat
pickup_lon
delivery_lat
delivery_lon
equipment
month_sin
month_cos
doy_sin
doy_cos
dow
lat_diff
lon_diff
```

## Local Validation Results

Chronological holdout performance:

| Metric | Result |
|---|---:|
| MAE | $YOUR_MAE |
| RMSE | $YOUR_RMSE |
| MAPE | YOUR_MAPE% |

These results are from the local chronological holdout and should not be interpreted as the final Spotter evaluation score.

The final evaluation metrics are calculated by Spotter after submission.

## Final Training

After validating the modeling approach, the LightGBM model was retrained using all available labeled training observations.

This final model was then used to generate predictions for the supplied validation dataset.

## Validation Predictions

The model generates predictions for all **12,000 validation loads**.

The final submission file contains:

```text
load_id,predicted_rate
```

Output:

```text
outputs/validation_predictions.csv
```

Predictions are matched to their corresponding `load_id`.

## December 2025 Prediction

The assessment also requires daily predictions for December 2025 using the fixed shipment configuration:

```text
Pickup: Lexington
Delivery: Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb
Period: December 1-31, 2025
```

The supplied scorer validates these predictions and generates:

```text
outputs/candidate_december.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/spotter-freight-rate-ml-assessment.git

cd spotter-freight-rate-ml-assessment
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

The solution uses:

```text
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.0,<3
lightgbm>=4.0,<5
scikit-learn>=1.4,<2
```

## Running the Notebook

Place the assessment datasets in your local input directory or attach them as Kaggle datasets.

Then run:

```text
notebook/freight_rate_prediction.ipynb
```

from top to bottom.

The notebook performs preprocessing, model training, evaluation, final fitting, prediction generation and output validation.

## Scorer

The supplied assessment scorer can be run using:

```bash
python score.py \
    --predictions validation_predictions.csv \
    --december-predictions december_chart_inputs.csv
```

Successful execution validates the required files and generates:

```text
scorer_results/candidate_december.png
```

## Reproducibility

A fixed random seed is used:

```python
RANDOM_STATE = 42
```

The same preprocessing and feature-engineering pipeline is applied consistently during training and inference.

## Notes

- Raw assessment data is intentionally excluded from the public repository.
- The validation predictions contain exactly one prediction for every supplied `load_id`.
- Local validation results are based on a chronological holdout.
- Final hidden-set performance is determined by Spotter after submission.

## Author

**Md. Raihan Siddik**

Machine Learning Assessment Submission
