from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def add_features(df, group_sizes):
    df = df.copy()

    df[["CabinDeck", "CabinNum", "CabinSide"]] = df["Cabin"].str.split("/", expand=True)
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

    df["TotalSpend"] = df[SPEND_COLS].fillna(0).sum(axis=1)
    df["AnySpend"] = df["TotalSpend"].gt(0).astype(int)

    pid = df["PassengerId"].str.split("_")
    group = pid.str[0]

    df["InGroupIndex"] = pd.to_numeric(pid.str[1], errors="coerce")
    df["GroupSize"] = group.map(group_sizes)

    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    all_groups = pd.concat([train["PassengerId"], test["PassengerId"]]).str.split("_").str[0]
    group_sizes = all_groups.value_counts()

    train = add_features(train, group_sizes)
    test = add_features(test, group_sizes)

    y = train.pop("Transported").astype(bool)

    drop_cols = ["PassengerId", "Cabin", "Name"]
    X = train.drop(columns=drop_cols)
    X_test = test.drop(columns=drop_cols)

    categorical = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    numeric = X.columns.difference(categorical).tolist()

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ]
    )

    model = make_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=350,
            min_samples_leaf=20,
            random_state=42,
        ),
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)
    print(f"Holdout accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")

    model.fit(X, y)
    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "Transported": model.predict(X_test).astype(bool),
        }
    )

    path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(path, index=False)

    print(f"Wrote {path}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()