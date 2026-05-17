import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 29
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CV_SEEDS = [13, 29, 42]
FOLDS = 5
EARLY_STOP = 200
MAX_ITER = 3000

# Set >0 to run Optuna before final CV; override with: python model.py --trials 50
OPTUNA_TRIALS = 0


def make_config(depth=6, learning_rate=0.10985745201142037, l2_leaf_reg=6.1652988679205905):
    return dict(
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        iterations=MAX_ITER,
    )


CONFIG = make_config()


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


def cv_score_config(X, y, cat_cols, config, seeds):
    scores = []
    for seed in seeds:
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
        for tr, va in cv.split(X, y):
            m = model(config, seed)
            m.fit(
                X.iloc[tr],
                y.iloc[tr],
                cat_features=cat_cols,
                eval_set=(X.iloc[va], y.iloc[va]),
                use_best_model=True,
                early_stopping_rounds=EARLY_STOP,
            )
            scores.append(accuracy_score(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


def cross_validate(X, y, cat_cols, config):
    scores, iters = [], []

    for seed in CV_SEEDS:
        fold_scores, fold_iters = [], []
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)

        for tr, va in cv.split(X, y):
            m = model(config, seed)
            m.fit(
                X.iloc[tr],
                y.iloc[tr],
                cat_features=cat_cols,
                eval_set=(X.iloc[va], y.iloc[va]),
                use_best_model=True,
                early_stopping_rounds=EARLY_STOP,
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


def tune_hyperparams(X, y, cat_cols, n_trials, tune_seeds):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        config = make_config(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
        )
        return cv_score_config(X, y, cat_cols, config, tune_seeds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    tuned = make_config(
        depth=best["depth"],
        learning_rate=best["learning_rate"],
        l2_leaf_reg=best["l2_leaf_reg"],
    )
    return tuned, study.best_value, study


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    stats = build_stats(train, test)
    train = add_features(train, stats)
    test = add_features(test, stats)
    return prepare(train, test), test


def main(trials=OPTUNA_TRIALS, tune_seeds=None):
    OUT.mkdir(exist_ok=True)
    tune_seeds = tune_seeds or [SEED]

    (X, y, X_test, cat_cols), test = load_data()
    config = make_config(**{k: CONFIG[k] for k in ("depth", "learning_rate", "l2_leaf_reg")})

    if trials > 0:
        print(f"Optuna: {trials} trials, {len(tune_seeds)} seed(s) x {FOLDS} folds per trial")
        baseline = cv_score_config(X, y, cat_cols, config, tune_seeds)
        print(f"Baseline CV ({config}): {baseline:.4f}")

        config, tuned_cv, study = tune_hyperparams(X, y, cat_cols, trials, tune_seeds)
        print(f"Optuna best CV: {tuned_cv:.4f}  params={config}")

        out = {
            "trials": trials,
            "tune_seeds": tune_seeds,
            "baseline_cv": baseline,
            "best_cv": tuned_cv,
            "best_params": {
                "depth": config["depth"],
                "learning_rate": config["learning_rate"],
                "l2_leaf_reg": config["l2_leaf_reg"],
            },
        }
        path_params = OUT / "optuna_best.json"
        path_params.write_text(json.dumps(out, indent=2))
        print(f"Wrote {path_params}")

    mean, std, best_iters = cross_validate(X, y, cat_cols, config)

    final_config = {**config, "iterations": best_iters}
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
    parser = argparse.ArgumentParser(description="Spaceship Titanic CatBoost pipeline")
    parser.add_argument(
        "--trials",
        type=int,
        default=OPTUNA_TRIALS,
        help="Optuna trials (0 = skip tuning, use CONFIG defaults)",
    )
    parser.add_argument(
        "--tune-seeds",
        type=int,
        nargs="+",
        default=[SEED],
        help="CV seeds during Optuna (default: 42 only, for speed)",
    )
    args = parser.parse_args()
    main(trials=args.trials, tune_seeds=args.tune_seeds)
