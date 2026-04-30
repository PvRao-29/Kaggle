"""Train a baseline model and write submission.csv for Spaceship Titanic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def engineer_features(df: pd.DataFrame, group_sizes: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    # Cabin → deck / number / side
    cabin = out["Cabin"].astype(str).str.split("/", expand=True)
    out["CabinDeck"] = cabin[0].replace("nan", np.nan)
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["CabinSide"] = cabin[2].replace("nan", np.nan)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["TotalSpend"] = out[spend_cols].fillna(0).sum(axis=1)
    out["AnySpend"] = (out[spend_cols].fillna(0).sum(axis=1) > 0).astype(int)

    gid = out["PassengerId"].str.split("_").str[0]
    out["GroupId"] = pd.to_numeric(gid, errors="coerce")
    out["InGroupIndex"] = pd.to_numeric(
        out["PassengerId"].str.split("_").str[1], errors="coerce"
    )

    if group_sizes is not None:
        out["GroupSize"] = gid.map(group_sizes)
    else:
        out["GroupSize"] = np.nan

    return out


def build_group_sizes(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    combined = pd.concat([train, test], axis=0)
    gid = combined["PassengerId"].str.split("_").str[0]
    return gid.value_counts()


def build_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = SimpleImputer(strategy="median")

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical),
            ("num", num_pipe, numeric),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    clf = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.06,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )
    return Pipeline(steps=[("prep", pre), ("clf", clf)])


def main() -> None:
    train_raw, test_raw = load_frames()
    group_sizes = build_group_sizes(train_raw, test_raw)

    train = engineer_features(train_raw, group_sizes)
    test = engineer_features(test_raw, group_sizes)

    y = train_raw["Transported"].astype(bool).values

    drop_cols = {"PassengerId", "Cabin", "Name", "Transported"}
    feature_cols = [c for c in train.columns if c not in drop_cols]

    categorical = [
        c
        for c in feature_cols
        if pd.api.types.is_object_dtype(train[c])
        or pd.api.types.is_bool_dtype(train[c])
    ]
    numeric = [c for c in feature_cols if c not in categorical]

    X = train[feature_cols]
    X_test = test[feature_cols]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(categorical, numeric)
    pipe.fit(X_tr, y_tr)
    val_pred = pipe.predict(X_val)
    print(f"Holdout accuracy: {accuracy_score(y_val, val_pred):.4f}")

    pipe.fit(X, y)
    test_pred = pipe.predict(X_test)

    submission = pd.DataFrame(
        {
            "PassengerId": test_raw["PassengerId"],
            "Transported": test_pred,
        }
    )
    out_path = Path(__file__).resolve().parent / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
