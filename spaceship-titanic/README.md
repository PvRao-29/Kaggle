# Spaceship Titanic

This repository trains a tabular ML model for the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition and generates a Kaggle-ready `submission.csv`.

The pipeline uses:
- Feature engineering from passenger/cabin IDs and onboard spending
- Mixed preprocessing for categorical and numeric features
- `HistGradientBoostingClassifier` from scikit-learn

## Model Approach

`model.py` follows this flow:

1. **Load data**
   - Reads `data/train.csv` and `data/test.csv`.

2. **Feature engineering (`add_features`)**
   - Splits `Cabin` into:
     - `CabinDeck`
     - `CabinNum` (numeric)
     - `CabinSide`
   - Creates spending features:
     - `TotalSpend` = sum of `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
     - `AnySpend` = whether `TotalSpend > 0`
   - Extracts group info from `PassengerId`:
     - `InGroupIndex` from the suffix after `_`
     - `GroupSize` computed across both train and test passenger groups

3. **Prepare target and columns**
   - Target: `Transported` converted to boolean
   - Drops leakage/identifier-style columns from training features:
     - `PassengerId`, `Cabin`, `Name`

4. **Preprocessing**
   - Categorical columns (`object`, `string`, `bool`):
     - `SimpleImputer(strategy="most_frequent")`
     - `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`
   - Numeric columns:
     - `SimpleImputer(strategy="median")`

5. **Model**
   - `HistGradientBoostingClassifier` with:
     - `max_depth=8`
     - `learning_rate=0.06`
     - `max_iter=350`
     - `min_samples_leaf=20`
     - `random_state=42`

6. **Validation + final training**
   - Stratified train/validation split
   - Prints holdout accuracy
   - Retrains on full training data

7. **Submission output**
   - Writes `outputs/submission.csv` with:
     - `PassengerId`
     - predicted `Transported` (bool)

## Requirements

- Python `>=3.12`
- Dependencies (from `pyproject.toml`):
  - `numpy`
  - `pandas`
  - `scikit-learn`


## Notes for Improvement

Potential next upgrades:
- Cross-validation instead of a single holdout split
- Better categorical handling (target encoding or CatBoost-style models)
- Additional domain features (family/group consistency, cabin interactions)
- Hyperparameter tuning (Optuna/grid search)
