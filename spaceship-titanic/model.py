from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 200

CATBOOST_CONFIGS = [
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 4000},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 4000},
    {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 8, "iterations": 5000},
    {"depth": 10, "learning_rate": 0.02, "l2_leaf_reg": 8, "iterations": 5000},
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3, "iterations": 3000},
]

def add_features(df, group_sizes):
    df = df.copy()

    df[["CabinDeck", "CabinNum", "CabinSide"]] = df["Cabin"].str.split("/", expand=True)
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

    groups = df["PassengerId"].str.split("_")
    group_id = groups.str[0]

    df["InGroupIndex"] = pd.to_numeric(groups.str[1], errors="coerce")
    df["GroupSize"] = group_id.map(group_sizes).fillna(1)

    df["TotalSpend"] = df[SPEND_COLS].fillna(0).sum(axis=1)
    df["AnySpend"] = (df["TotalSpend"] > 0).astype(int)

    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["SpendPerGroupMember"] = df["TotalSpend"] / df["GroupSize"].replace(0, np.nan)
    df["CryoSleepAnySpendMismatch"] = (
        (df["CryoSleep"] == True) & (df["AnySpend"] == 1)  # noqa: E712
    ).astype(int)
    df["DeckSide"] = (
        df["CabinDeck"].fillna("Unknown") + "_" + df["CabinSide"].fillna("Unknown")
    )
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["VIPMissing"] = df["VIP"].isna().astype(int)

    return df


def catboost_model(config):
    return CatBoostClassifier(
        depth=config["depth"],
        learning_rate=config["learning_rate"],
        iterations=config["iterations"],
        l2_leaf_reg=config["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
    )


def prepare_catboost_data(X, X_test):
    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    X_cb = X.copy()
    X_test_cb = X_test.copy()

    for col in cat_cols:
        X_cb[col] = X_cb[col].fillna("Missing").astype(str)
        X_test_cb[col] = X_test_cb[col].fillna("Missing").astype(str)

    return X_cb, X_test_cb, cat_cols


def evaluate_config(X, y, cat_cols, config):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    best_iterations = []
    for train_idx, val_idx in cv.split(X, y):
        fold_model = catboost_model(config)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fold_model.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        y_pred = fold_model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
        best_iterations.append(max(1, fold_model.get_best_iteration()))

    score_array = np.array(scores)
    return {
        "config": config,
        "cv_mean": float(score_array.mean()),
        "cv_std": float(score_array.std()),
        "best_iterations": int(np.median(best_iterations)),
        "stability_score": float(score_array.mean() - 0.5 * score_array.std()),
    }


def tune_configs(X, y, cat_cols):
    results = []
    for i, config in enumerate(CATBOOST_CONFIGS, start=1):
        result = evaluate_config(X, y, cat_cols, config)
        results.append(result)
        print(
            f"[{i}/{len(CATBOOST_CONFIGS)}] "
            f"depth={config['depth']} lr={config['learning_rate']} "
            f"l2={config['l2_leaf_reg']} -> "
            f"cv={result['cv_mean']:.4f} (+/- {result['cv_std']:.4f}), "
            f"best_iters={result['best_iterations']}"
        )

    return sorted(results, key=lambda r: (r["stability_score"], r["cv_mean"]), reverse=True)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    all_ids = pd.concat([train["PassengerId"], test["PassengerId"]])
    group_sizes = all_ids.str.split("_").str[0].value_counts()

    train = add_features(train, group_sizes)
    test = add_features(test, group_sizes)

    y = train.pop("Transported").astype(bool)

    drop_cols = ["PassengerId", "Cabin", "Name"]
    X = train.drop(columns=drop_cols)
    X_test = test.drop(columns=drop_cols)

    X_cb, X_test_cb, cat_cols = prepare_catboost_data(X, X_test)
    ranked = tune_configs(X_cb, y, cat_cols)
    best = ranked[0]
    chosen_config = {**best["config"], "iterations": best["best_iterations"]}
    model = catboost_model(chosen_config)

    print(
        "Chosen config: "
        f"depth={chosen_config['depth']} lr={chosen_config['learning_rate']} "
        f"l2={chosen_config['l2_leaf_reg']} iterations={chosen_config['iterations']}"
    )
    print(f"Chosen CV accuracy: {best['cv_mean']:.4f} (+/- {best['cv_std']:.4f})")

    model.fit(X_cb, y, cat_features=cat_cols)
    preds = model.predict(X_test_cb)

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "Transported": preds.astype(bool),
        }
    )

    path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(path, index=False)

    print(f"Wrote {path}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()