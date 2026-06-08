"""
Dixon-Coles (1997) adaptált Poisson modell.
Csapatonkénti attack/defense paramétert becsül MLE-vel,
majd Poisson-eloszlással számolja a meccs valószínűségeket.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "dc_model.pkl")


class DixonColesModel:
    """
    Simplified Dixon-Coles Poisson model.
    - attack[team]: csapat támadóereje (log-skálán)
    - defense[team]: csapat védelmi ereje (log-skálán, alacsonyabb = jobb)
    - home_adv: hazai pályaelőny
    - rho: alacsony gólszámoknál korrekció (0-0, 1-0, 0-1, 1-1)
    """

    def __init__(self):
        self.attack = {}
        self.defense = {}
        self.home_adv = 0.25
        self.rho = -0.1
        self.teams = []
        self._fitted = False

    @staticmethod
    def _tau(x, y, mu, nu, rho):
        """Dixon-Coles korrekció alacsony gólszámoknál."""
        if x == 0 and y == 0:
            return 1 - mu * nu * rho
        elif x == 0 and y == 1:
            return 1 + mu * rho
        elif x == 1 and y == 0:
            return 1 + nu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _neg_log_lik(self, params, matches, team_idx):
        n = len(self.teams)
        attack = params[:n]
        defense = params[n:2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]

        ll = 0.0
        for h, a, hg, ag in matches:
            hi = team_idx[h]
            ai = team_idx[a]
            mu = np.exp(attack[hi] + defense[ai] + home_adv)
            nu = np.exp(attack[ai] + defense[hi])
            tau = self._tau(hg, ag, mu, nu, rho)
            if tau <= 0:
                return 1e10
            ll += (np.log(max(tau, 1e-10))
                   + poisson.logpmf(hg, mu)
                   + poisson.logpmf(ag, nu))
        return -ll

    def fit(self, df: pd.DataFrame, recent_n: int = 2500):
        """
        Betanítás a legutóbbi `recent_n` meccs adatain.
        Újabb adatokra fókuszálunk (relevánsabb a jelenlegi erőviszonyokhoz).
        """
        df = df.sort_values("date").tail(recent_n).reset_index(drop=True)
        self.teams = sorted(
            set(df["home_team"].tolist() + df["away_team"].tolist())
        )
        team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        matches = [
            (row["home_team"], row["away_team"],
             int(row["home_goals"]), int(row["away_goals"]))
            for _, row in df.iterrows()
        ]

        x0 = np.concatenate([
            np.zeros(n),  # attack
            np.zeros(n),  # defense
            [0.25],       # home_adv
            [-0.1],       # rho
        ])

        print("  Dixon-Coles modell tanítása...")
        res = minimize(
            self._neg_log_lik,
            x0,
            args=(matches, team_idx),
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-6},
        )

        self.attack = dict(zip(self.teams, res.x[:n]))
        self.defense = dict(zip(self.teams, res.x[n:2 * n]))
        self.home_adv = float(res.x[2 * n])
        self.rho = float(res.x[2 * n + 1])
        self._fitted = True
        print(f"  Dixon-Coles kész (home_adv={self.home_adv:.3f}, rho={self.rho:.3f})")
        return self

    def predict_proba(self, home_team: str, away_team: str, max_goals: int = 8):
        """
        Visszatér: np.array([P_hazai_győzelem, P_döntetlen, P_vendég_győzelem])
        """
        if not self._fitted:
            return np.array([0.45, 0.27, 0.28])

        avg_att = float(np.mean(list(self.attack.values()))) if self.attack else 0.0
        avg_def = float(np.mean(list(self.defense.values()))) if self.defense else 0.0

        h_att = self.attack.get(home_team, avg_att)
        a_att = self.attack.get(away_team, avg_att)
        h_def = self.defense.get(home_team, avg_def)
        a_def = self.defense.get(away_team, avg_def)

        mu = np.exp(h_att + a_def + self.home_adv)  # várható hazai gólok
        nu = np.exp(a_att + h_def)                   # várható vendég gólok

        # Együttes valószínűségi mátrix (i hazai gól, j vendég gól)
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = self._tau(i, j, mu, nu, self.rho)
                probs[i, j] = max(tau, 0) * poisson.pmf(i, mu) * poisson.pmf(j, nu)

        total = probs.sum()
        if total <= 0:
            return np.array([0.45, 0.27, 0.28])
        probs /= total

        p_h = float(np.sum(np.tril(probs, -1)))  # hazai győzelem: i > j
        p_d = float(np.trace(probs))              # döntetlen: i == j
        p_a = float(np.sum(np.triu(probs, 1)))    # vendég győzelem: i < j

        return np.array([p_h, p_d, p_a])


def save_dc_model(model: DixonColesModel):
    with open(DC_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_dc_model():
    if not os.path.exists(DC_MODEL_PATH):
        return None
    with open(DC_MODEL_PATH, "rb") as f:
        return pickle.load(f)
