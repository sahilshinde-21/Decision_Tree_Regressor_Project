# Used Car Price Prediction 🚗

A beginner ML project that predicts the resale price of used cars using a
**Decision Tree Regressor**, trained on real-world scraped listing data from
Quikr (via Kaggle).

## Problem

Given a car's name, company, year, kilometers driven, and fuel type, predict
its resale price. The raw dataset is messy — inconsistent formatting, missing
values, and placeholder text instead of numbers — so a large part of this
project is data cleaning, not just modeling.

## Dataset

- Source: [Quikr Cars dataset](https://www.kaggle.com/datasets) (Kaggle)
- 892 raw listings, 719 usable after cleaning
- Features: `name`, `company`, `year`, `kms_driven`, `fuel_type`
- Target: `Price`

## Data Cleaning

The raw data had several issues that had to be fixed before modeling:
- `year` contained non-numeric junk values (e.g. `'TOUR'`, `'150k'`)
- `Price` had comma-formatted numbers and 87 rows with `"Ask For Price"`
  instead of an actual value
- `kms_driven` had text (`"kms"`) and commas mixed into the numbers, plus
  missing values
- `fuel_type` had missing values
- `name` was too specific per row (nearly unique), so it was trimmed to the
  first 3 words to group similar car variants together
- 94 duplicate rows were removed
- A handful of extreme outlier prices were filtered out

## Approach

1. Clean and prepare the raw CSV
2. Explore the data visually (price by company, year, kms driven, fuel type)
3. One-hot encode categorical columns (`name`, `company`, `fuel_type`)
4. Train/test split (80/20)
5. Train a `DecisionTreeRegressor`, manually tuning `max_depth` by sweeping
   values from 2–15 and comparing train vs test R² to find where overfitting
   starts (no `GridSearchCV` used — kept intentionally simple)
6. Evaluate the final model on held-out test data

## Results

| Metric | Value |
|---|---|
| R² | 0.506 |
| MAE | ₹1,75,157 |
| RMSE | ₹2,80,311 |

The model explains roughly half the variance in used car prices. Error is
still meaningful (~₹1.75 lakh average), so this isn't production-ready
pricing — it's a first pass at understanding the relationship between a
car's basic attributes and its resale value.

## Limitations

- Only 719 rows remained after cleaning — small for 25+ car brands
- Only `max_depth` was tuned manually; other hyperparameters
  (`min_samples_leaf`, `criterion`, etc.) were left at default since grid
  search hasn't been covered yet
- One-hot encoding on `name`/`company` creates many sparse columns, which
  the model leans on heavily — a different feature engineering approach
  might generalize better

## Tools Used

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook DecisionTreeRegressor_Project.ipynb
```
