# Spaceship Titanic

**CURRENTLY ACHIEVING A SCORE IN THE TOP 200 OUT OF 140,000 ENTRANTS**

This repository trains a tabular ML model for the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition and generates a Kaggle-ready `submission.csv`.

The pipeline uses:
- Feature engineering from passenger/cabin IDs, family/group structure, and onboard spending
- Group-aware imputations and interaction features
- `CatBoostClassifier` with repeated stratified cross-validation

## Model Approach

`model.py` follows this flow:

1. **Load data**
   - Reads `data/train.csv` and `data/test.csv`.

2. **Build dataset-level statistics (`build_stats`)**
   - Computes pooled train+test lookup stats used for feature construction and imputations:
     - `group_sizes`
     - `family_sizes` (surname counts)
     - `cabin_counts`
     - group-level `HomePlanet` / `Destination` mode
     - group-level median age

3. **Feature engineering (`add_features`)**
   - Splits `Cabin` into:
     - `CabinDeck`
     - `CabinNum` (numeric)
     - `CabinSide`
   - Adds group/family features:
     - `GroupId`, `InGroupIndex`, `GroupSize`
     - `Surname`, `FamilySize`
     - `CabinOccupancy`
   - Fills key missing values using group-level signals then global defaults:
     - `HomePlanet`, `Destination`, `Age`
   - Creates spending and behavior features:
     - `TotalSpend` = sum of `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
     - `AnySpend`, `LuxurySpend`, `EssentialSpend`
     - `SpendPerGroupMember`, `LuxuryToEssential`
     - `AwakeAdultZeroSpend`, `CryoSleepInferredFromSpend`
   - Adds cabin/age interactions:
     - `DeckSide`, `DeckAnySpend`, `CabinRegion`, `CabinKnown`
     - `IsAlone`, `IsChild`, `GroupAvgAge`
     - missingness indicators (`AgeMissing`, `VIPMissing`)

4. **Prepare target and columns (`prepare`)**
   - Target: `Transported` converted to boolean
   - Drops identifier-style columns:
     - `PassengerId`, `Cabin`, `Name`, `GroupId`
   - Converts boolean-like columns for CatBoost compatibility:
     - `CryoSleep`, `VIP` -> `1.0` / `0.0`
   - Keeps string/object columns as categorical features for CatBoost

5. **Model (`model`)**
   - `CatBoostClassifier` with:
     - `depth=6`
     - `learning_rate=0.05`
     - `l2_leaf_reg=3`
     - `iterations=3000` (used during CV with early stopping)
     - `loss_function="Logloss"`
     - `eval_metric="Accuracy"`
     - `auto_class_weights="Balanced"`

6. **Validation (`cross_validate`)**
   - Repeated stratified K-fold:
     - Seeds: `13`, `29`, `42`
     - Folds per seed: `5` (`3 x 5 = 15` validation runs total)
   - Uses early stopping (`early_stopping_rounds=200`)
   - Collects best iteration per fold and uses median best iteration for final training

7. **Final training + submission output**
   - Trains final model on full training set with selected iteration count
   - Writes `outputs/submission.csv` with:
     - `PassengerId`
     - predicted `Transported` (bool)

## Requirements

- Python `>=3.12`
- Dependencies (from `pyproject.toml`):
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `catboost`
