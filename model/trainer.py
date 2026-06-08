"""
v4: XGBoost + LightGBM ensemble + Optuna hyperparameter tuning + Dixon-Coles
Új feature-ök: H2H statisztikák, fáradtság (days_rest)
"""
import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "encoder.pkl")

FEATURES = [
    # Hazai csapat - hazai/vendég szeparált forma
    "home_home_gf", "home_home_ga", "home_home_form", "home_home_n",
    "home_away_gf", "home_away_ga", "home_away_form", "home_away_n",
    "home_overall_form", "home_overall_gf", "home_overall_ga", "home_momentum",
    # Vendég csapat - hazai/vendég szeparált forma
    "away_home_gf", "away_home_ga", "away_home_form", "away_home_n",
    "away_away_gf", "away_away_ga", "away_away_form", "away_away_n",
    "away_overall_form", "away_overall_gf", "away_overall_ga", "away_momentum",
    # Elo + streak
    "home_elo", "away_elo", "elo_diff",
    "home_streak", "away_streak",
    # H2H statisztikák
    "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_avg_goals", "h2h_n",
    # Fáradtság
    "home_days_rest", "away_days_rest",
]


def _optuna_tune(X_train, y_train, n_trials: int = 20):
    """
    Optuna alapú hyperparameter keresés XGBoost és LightGBM modellekhez.
    n_trials=20: kb. 3-5 perc futásidő.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=3, shuffle=False)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        }
        model = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="accuracy", n_jobs=-1)
        return scores.mean()

    print("  Optuna: XGBoost hyperparameter keresés...")
    study_xgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_xgb.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGBoost legjobb CV accuracy: {study_xgb.best_value:.3f}")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    def objective_lgbm(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        }
        model = LGBMClassifier(**params, random_state=42, verbose=-1)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="accuracy", n_jobs=-1)
        return scores.mean()

    print("  Optuna: LightGBM hyperparameter keresés...")
    study_lgbm = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_lgbm.optimize(objective_lgbm, n_trials=n_trials, show_progress_bar=False)
    print(f"  LightGBM legjobb CV accuracy: {study_lgbm.best_value:.3f}")

    return study_xgb.best_params, study_lgbm.best_params


class EnsembleModel:
    """XGBoost + LightGBM súlyozott ensemble, kalibrált valószínűségekkel."""

    def __init__(self, xgb_params: dict = None, lgbm_params: dict = None,
                 xgb_weight: float = 0.5, lgbm_weight: float = 0.5):
        self.xgb_params = xgb_params or {}
        self.lgbm_params = lgbm_params or {}
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.xgb_cal = None
        self.lgbm_cal = None
        self.classes_ = None

    def fit(self, X, y):
        default_xgb = dict(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1,
            use_label_encoder=False, eval_metric="mlogloss", random_state=42,
        )
        default_lgbm = dict(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
        )
        xgb_cfg = {**default_xgb, **self.xgb_params}
        lgbm_cfg = {**default_lgbm, **self.lgbm_params}

        # Eltávolítjuk a nem-XGBoost paramétereket, ha keverednek
        for k in ["use_label_encoder", "eval_metric"]:
            lgbm_cfg.pop(k, None)

        xgb = XGBClassifier(**xgb_cfg)
        lgbm = LGBMClassifier(**lgbm_cfg)

        self.xgb_cal = CalibratedClassifierCV(xgb, cv=3, method="isotonic")
        self.lgbm_cal = CalibratedClassifierCV(lgbm, cv=3, method="isotonic")

        self.xgb_cal.fit(X, y)
        self.lgbm_cal.fit(X, y)
        self.classes_ = self.xgb_cal.classes_
        return self

    def predict_proba(self, X):
        p_xgb = self.xgb_cal.predict_proba(X)
        p_lgbm = self.lgbm_cal.predict_proba(X)
        return self.xgb_weight * p_xgb + self.lgbm_weight * p_lgbm

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def train(df: pd.DataFrame, optuna_trials: int = 8) -> tuple:
    from model.dixon_coles import DixonColesModel, save_dc_model

    df_clean = df.dropna(subset=FEATURES + ["result"])
    df_clean = df_clean[
        (df_clean["home_home_n"] >= 2) | (df_clean["home_away_n"] >= 2)
    ].copy()

    if len(df_clean) < 100:
        raise ValueError(f"Kevés adat: {len(df_clean)} meccs")

    X = df_clean[FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(df_clean["result"].values)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Tanítás: {len(X_train)} meccs | Teszt: {len(X_test)} meccs")

    # ── Optuna hyperparameter tuning ─────────────────────────────────────────
    xgb_params, lgbm_params = _optuna_tune(X_train, y_train, n_trials=optuna_trials)

    # ── Ensemble tanítás ─────────────────────────────────────────────────────
    print("  Ensemble modell tanítása (kalibrációval)...")
    model = EnsembleModel(xgb_params=xgb_params, lgbm_params=lgbm_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Ensemble pontossága (test): {acc:.1%}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print("  ML modell (v4) elmentve.")

    # ── Dixon-Coles modell ───────────────────────────────────────────────────
    # A raw DataFrame-t a train() hívója adja át - de ez a processed_df.
    # A DC modellt a raw_df-en tanítjuk (goals kell), amit az app.py-ból kapunk.
    # Ezért a DC tanítás az app.py-ban fut, nem itt.
    # (Az app.py külön hívja a dc_model.fit(raw_df)-et)

    return model, le


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le
