"""Preprocessing for the Titanic dataset.

Idea: we split the data FIRST, then learn every "fitted" number (imputation
medians, the mode, the scaler) from the TRAIN split only and apply those same
numbers to validation/test. That keeps it leak-free without any custom classes.

The learned numbers are bundled in a plain dict `prep` so the Streamlit app can
load it later and preprocess new data exactly the same way.
"""
import os
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# columns we keep as-is (numeric / 0-1) and columns we one-hot encode
BASE_COLS = ["Age", "Fare", "Pclass", "HasCabin"]
CAT_COLS = ["Sex", "Embarked", "TitleGrp", "FamilyBin"]
SCALE_COLS = ["Age", "Fare", "Pclass"]   # continuous-ish -> standardize for the NN


def add_features(df):
    """Add engineered columns. Per-row only, so it's safe to run on any data."""
    df = df.copy()

    # Title from Name (EDA: captures sex + age + status in one feature)
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["TitleGrp"] = df["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
    df["TitleGrp"] = df["TitleGrp"].where(
        df["TitleGrp"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")

    # family size -> bins (EDA: survival is non-monotonic in family size)
    family = df["SibSp"] + df["Parch"] + 1
    df["FamilyBin"] = pd.cut(family, [0, 1, 4, 100], labels=["Alone", "Small", "Large"])

    # Cabin is 77% missing -> just a yes/no flag (EDA: 0.67 vs 0.30 survival)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def _build_features(df, prep, fit):
    """Impute + log-fare + one-hot, using the numbers stored in `prep`.

    fit=True  -> we're on the train split: also record the final column list.
    fit=False -> align columns to the train ones (handles unseen/missing values).
    """
    df = add_features(df)

    # impute with values that were learned on train (see `preprocess`)
    df["Age"] = df["Age"].fillna(df["TitleGrp"].map(prep["age_medians"])).fillna(prep["age_global"])
    df["Embarked"] = df["Embarked"].fillna(prep["embarked_mode"])
    df["Fare"] = df["Fare"].fillna(prep["fare_median"])

    # Fare is very skewed -> log makes it well-behaved
    df["Fare"] = np.log1p(df["Fare"])

    # keep only the columns we use, then one-hot the categoricals
    feats = pd.get_dummies(df[BASE_COLS + CAT_COLS], columns=CAT_COLS)

    if fit:
        prep["columns"] = feats.columns.tolist()
    else:
        # add any missing dummy columns as 0, drop any unexpected ones, keep order
        feats = feats.reindex(columns=prep["columns"], fill_value=0)

    return feats.astype("float32")


def preprocess(df, seed=42):
    """Split -> learn numbers on train -> transform all. Returns arrays + prep dict."""
    df = add_features(df)
    y = df["Survived"].astype("float32").to_numpy()

    # stratified 60/20/20 split (stratify because classes are imbalanced ~38%)
    df_tr, df_te, y_tr, y_te = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=seed)
    df_tr, df_va, y_tr, y_va = train_test_split(
        df_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=seed)  # 0.25*0.8 = 0.2

    # learn all "fitted" numbers from TRAIN only
    prep = {
        "age_medians": df_tr.groupby("TitleGrp")["Age"].median().to_dict(),
        "age_global": df_tr["Age"].median(),
        "embarked_mode": df_tr["Embarked"].mode()[0],
        "fare_median": df_tr["Fare"].median(),
    }

    # build feature tables (fit on train, align val/test to train columns)
    X_tr = _build_features(df_tr, prep, fit=True)
    X_va = _build_features(df_va, prep, fit=False)
    X_te = _build_features(df_te, prep, fit=False)

    # scale the continuous columns (fit scaler on train, apply to all)
    scaler = StandardScaler().fit(X_tr[SCALE_COLS])
    for X in (X_tr, X_va, X_te):
        X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])
    prep["scaler"] = scaler

    return (X_tr.to_numpy(), X_va.to_numpy(), X_te.to_numpy(),
            y_tr, y_va, y_te, prep)


def transform(df, prep):
    """Preprocess new data with an already-learned `prep` (used by the app)."""
    feats = _build_features(df, prep, fit=False)
    feats[SCALE_COLS] = prep["scaler"].transform(feats[SCALE_COLS])
    return feats.to_numpy()


if __name__ == "__main__":
    os.environ[
        "KAGGLE_API_TOKEN"] = "KGAT_ec71a988f9111fa61c3453b31f5f2334"  # the KGAT_... value, after revoking the exposed one

    path = kagglehub.competition_download("titanic")
    print("Path to competition files:", path)

    df = pd.read_csv(f"{path}/train.csv")

    X_tr, X_va, X_te, y_tr, y_va, y_te, prep = preprocess(df)
    print("features:", prep["columns"])
    print("shapes:", X_tr.shape, X_va.shape, X_te.shape)
    print("positive rate  train=%.3f val=%.3f test=%.3f"
          % (y_tr.mean(), y_va.mean(), y_te.mean()))
    print("NaNs in train?", np.isnan(X_tr).any())