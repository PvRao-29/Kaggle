from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 42
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CV_SEEDS = [13, 29, 42]
FOLDS = 5

CONFIG = dict(depth=6, learning_rate=0.05, l2_leaf_reg=3, iterations=3000)


def mode_or_nan(s):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else np.nan


def build_stats(train, test):
    all_df = pd.concat([train, test], ignore_index=True)
    group = all_df["PassengerId"].str.split("_").str[0]
    surname = all_df["Name"].fillna("").str.split().str[-1].replace("", "Unknown")

    return {
        "group_sizes": group.value_counts(),
        "family_sizes": surname.value_counts(),
        "cabin_counts": all_df["Cabin"].fillna("UnknownCabin").value_counts(),
        "group_home": all_df.assign(GroupId=group).groupby("GroupId")["HomePlanet"].agg(mode_or_nan),
        "group_dest": all_df.assign(GroupId=group).groupby("GroupId")["Destination"].agg(mode_or_nan),
        "group_age": all_df.assign(GroupId=group).groupby("GroupId")["Age"].median(),
        "home": train["HomePlanet"].mode().iat[0],
        "dest": train["Destination"].mode().iat[0],
        "age": train["Age"].median(),
    }


def add_features(df, s):
    df = df.copy()
    df[SPEND] = df[SPEND].fillna(0)

    spend = df[SPEND].sum(axis=1)
    missing_cryo = df["CryoSleep"].isna()

    df["CryoSleepInferredFromSpend"] = (missing_cryo & spend.gt(0)).astype(int)
    df.loc[missing_cryo & spend.gt(0), "CryoSleep"] = False
    df.loc[df["CryoSleep"].eq(True), SPEND] = 0

    df[["CabinDeck", "CabinNum", "CabinSide"]] = df["Cabin"].str.split("/", expand=True)
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

    ids = df["PassengerId"].str.split("_")
    group = ids.str[0]
    surname = df["Name"].fillna("").str.split().str[-1].replace("", "Unknown")

    df["GroupId"] = group
    df["InGroupIndex"] = pd.to_numeric(ids.str[1], errors="coerce")
    df["GroupSize"] = group.map(s["group_sizes"]).fillna(1)
    df["Surname"] = surname
    df["FamilySize"] = surname.map(s["family_sizes"]).fillna(1)
    df["CabinOccupancy"] = df["Cabin"].fillna("UnknownCabin").map(s["cabin_counts"]).fillna(1)

    df["HomePlanet"] = df["HomePlanet"].fillna(group.map(s["group_home"])).fillna(s["home"])
    df["Destination"] = df["Destination"].fillna(group.map(s["group_dest"])).fillna(s["dest"])
    df["Age"] = df["Age"].fillna(group.map(s["group_age"])).fillna(s["age"])

    df["TotalSpend"] = df[SPEND].sum(axis=1)
    df["AnySpend"] = df["TotalSpend"].gt(0).astype(int)
    df["IsAlone"] = df["GroupSize"].eq(1).astype(int)
    df["IsChild"] = df["Age"].lt(13).astype(int)
    df["AwakeAdultZeroSpend"] = (
        df["Age"].ge(18) & df["CryoSleep"].eq(False) & df["TotalSpend"].eq(0)
    ).astype(int)
    df["SpendPerGroupMember"] = df["TotalSpend"] / df["GroupSize"]
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"] + df["FoodCourt"]
    df["EssentialSpend"] = df["RoomService"] + df["ShoppingMall"]
    df["LuxuryToEssential"] = df["LuxurySpend"] / (df["EssentialSpend"] + 1)
    df["GroupAvgAge"] = group.map(s["group_age"]).fillna(s["age"])

    deck = df["CabinDeck"].fillna("Unknown")
    side = df["CabinSide"].fillna("Unknown")

    df["DeckSide"] = deck + "_" + side
    df["DeckAnySpend"] = deck + "_" + df["AnySpend"].astype(str)
    df["CabinRegion"] = (
        pd.cut(
            df["CabinNum"],
            bins=[-np.inf, 300, 600, 900, 1200, np.inf],
            labels=["Early", "MidEarly", "Mid", "MidLate", "Late"],
        )
        .astype("object")
        .fillna("Unknown")
    )
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["VIPMissing"] = df["VIP"].isna().astype(int)

    return df


def prepare(train, test):
    y = train.pop("Transported").astype(bool)

    drop = ["PassengerId", "Cabin", "Name", "GroupId"]
    X = train.drop(columns=drop)
    X_test = test.drop(columns=drop)

    for col in ["CryoSleep", "VIP"]:
        X[col] = X[col].map({True: 1.0, False: 0.0})
        X_test[col] = X_test[col].map({True: 1.0, False: 0.0})

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)

    return X, y, X_test, cat_cols


def model(config, seed):
    return CatBoostClassifier(
        **config,
        loss_function="Logloss",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
    )


def cross_validate(X, y, cat_cols):
    scores, iters = [], []

    for seed in CV_SEEDS:
        fold_scores, fold_iters = [], []
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)

        for tr, va in cv.split(X, y):
            m = model(CONFIG, seed)
            m.fit(
                X.iloc[tr],
                y.iloc[tr],
                cat_features=cat_cols,
                eval_set=(X.iloc[va], y.iloc[va]),
                use_best_model=True,
                early_stopping_rounds=200,
            )

            fold_scores.append(accuracy_score(y.iloc[va], m.predict(X.iloc[va])))
            fold_iters.append(max(1, m.get_best_iteration()))

        scores += fold_scores
        iters += fold_iters

        print(
            f"seed={seed} CV accuracy: {np.mean(fold_scores):.4f} "
            f"(+/- {np.std(fold_scores):.4f}), best_iters={int(np.median(fold_iters))}"
        )

    return np.mean(scores), np.std(scores), int(np.median(iters))


def main():
    OUT.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    stats = build_stats(train, test)
    train = add_features(train, stats)
    test = add_features(test, stats)

    X, y, X_test, cat_cols = prepare(train, test)

    mean, std, best_iters = cross_validate(X, y, cat_cols)

    final_config = {**CONFIG, "iterations": best_iters}
    final_model = model(final_config, SEED)
    final_model.fit(X, y, cat_features=cat_cols)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Transported": final_model.predict(X_test).astype(bool),
    })

    path = OUT / "submission.csv"
    submission.to_csv(path, index=False)

    importances = pd.Series(
        final_model.get_feature_importance(),
        index=X.columns,
    ).sort_values(ascending=False)

    print(
        f"Using config: depth={final_config['depth']} "
        f"lr={final_config['learning_rate']} "
        f"l2={final_config['l2_leaf_reg']} "
        f"iterations={final_config['iterations']}"
    )
    print(f"Repeated CV accuracy: {mean:.4f} (+/- {std:.4f}) over {len(CV_SEEDS)}x{FOLDS} folds")
    print("Top feature importances:")
    print(importances.head(12).round(3).to_string())
    print(f"Wrote {path}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()