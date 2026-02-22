"""
Morocco Génération Green — Model Library
Implements: Holt-Winters, ARIMA(lite), Random Forest, SVR, Polynomial Trend, Ensemble
All models expose a common fit/predict interface for adaptive model selection.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import detrend
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
try:
    from statsmodels.tsa.api import VAR as VARMODEL
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def evaluate(y_true, y_pred, name="model"):
    return {
        "Model": name,
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE (%)": round(mape(y_true, y_pred), 4),
        "R²": round(r2(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────────────────────
# 1. Holt-Winters Double Exponential Smoothing (trend-aware)
# ─────────────────────────────────────────────────────────────

class HoltWinters:
    """Holt-Winters double exponential smoothing with trend dampening."""

    def __init__(self, alpha=None, beta=None, phi=0.98):
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self._level = None
        self._trend = None

    def _init_params(self, y):
        self._level = y[0]
        self._trend = y[1] - y[0] if len(y) > 1 else 0

    def _smooth(self, y, alpha, beta, phi):
        n = len(y)
        levels = np.zeros(n)
        trends = np.zeros(n)
        levels[0] = y[0]
        trends[0] = y[1] - y[0] if n > 1 else 0
        for t in range(1, n):
            prev_l = levels[t - 1]
            prev_b = trends[t - 1]
            levels[t] = alpha * y[t] + (1 - alpha) * (prev_l + phi * prev_b)
            trends[t] = beta * (levels[t] - prev_l) + (1 - beta) * phi * prev_b
        return levels, trends

    def _loss(self, params, y):
        alpha, beta = params
        if not (0 < alpha < 1 and 0 < beta < 1):
            return 1e10
        levels, trends = self._smooth(y, alpha, beta, self.phi)
        fitted = levels[:-1] + self.phi * trends[:-1]
        return np.mean((y[1:] - fitted) ** 2)

    def fit(self, y):
        y = np.array(y, dtype=float)
        if self.alpha is None or self.beta is None:
            res = minimize(self._loss, [0.3, 0.1], args=(y,),
                           bounds=[(0.01, 0.99), (0.01, 0.99)], method="L-BFGS-B")
            self.alpha, self.beta = res.x
        levels, trends = self._smooth(y, self.alpha, self.beta, self.phi)
        self._level = levels[-1]
        self._trend = trends[-1]
        self._fitted = np.concatenate([[y[0]], levels[:-1] + self.phi * trends[:-1]])
        return self

    def predict(self, h):
        forecasts = []
        l, b = self._level, self._trend
        for k in range(1, h + 1):
            phi_sum = sum(self.phi ** j for j in range(1, k + 1))
            forecasts.append(l + phi_sum * b)
        return np.array(forecasts)

    def fitted_values(self):
        return self._fitted


# ─────────────────────────────────────────────────────────────
# 2. ARIMA Lite (AR + differencing, MLE estimation)
# ─────────────────────────────────────────────────────────────

class ARIMALite:
    """AR(p) model on d-th differenced series."""

    def __init__(self, p=2, d=1):
        self.p = p
        self.d = d
        self.coeffs = None
        self.intercept = None
        self._orig = None
        self._diff = None

    def _difference(self, y, d):
        for _ in range(d):
            y = np.diff(y)
        return y

    def _undifference(self, original, forecasts, d):
        result = list(forecasts)
        for _ in range(d):
            last = original[-1] if len(result) == len(forecasts) else result[-1]
            result = [last + r for r in result]
            result = np.cumsum([original[-(d)] ] + result[:-1]).tolist()
        return np.array(result)

    def fit(self, y):
        y = np.array(y, dtype=float)
        self._orig = y.copy()
        dy = self._difference(y, self.d)
        self._diff = dy.copy()
        n = len(dy)
        if n <= self.p:
            self.p = max(1, n - 2)
        X = np.column_stack([dy[i:n - self.p + i] for i in range(self.p)])
        y_ar = dy[self.p:]
        model = Ridge(alpha=0.1)
        model.fit(X, y_ar)
        self.coeffs = model.coef_
        self.intercept = model.intercept_
        # fitted values in original space
        fitted_diff = model.predict(X)
        self._last_p = dy[-self.p:]
        return self

    def predict(self, h):
        buf = list(self._last_p)
        preds_diff = []
        for _ in range(h):
            x = np.array(buf[-self.p:])
            val = self.intercept + np.dot(self.coeffs, x)
            preds_diff.append(val)
            buf.append(val)
        # Undo differencing
        last_vals = list(self._orig[-self.d:]) if self.d > 0 else list(self._orig[-1:])
        result = []
        carry = self._orig[-1]
        for v in preds_diff:
            carry = carry + v
            result.append(carry)
        return np.array(result)


# ─────────────────────────────────────────────────────────────
# 3. Polynomial Trend Extrapolation
# ─────────────────────────────────────────────────────────────

class PolynomialTrend:
    def __init__(self, degree=2):
        self.degree = degree
        self.coeffs = None
        self._n = None

    def fit(self, y):
        y = np.array(y, dtype=float)
        self._n = len(y)
        t = np.arange(self._n)
        self.coeffs = np.polyfit(t, y, self.degree)
        self._poly = np.poly1d(self.coeffs)
        return self

    def predict(self, h):
        t_future = np.arange(self._n, self._n + h)
        return self._poly(t_future)

    def fitted_values(self):
        return self._poly(np.arange(self._n))


# ─────────────────────────────────────────────────────────────
# 4. Random Forest Regressor (with lag features)
# ─────────────────────────────────────────────────────────────

class RFForecaster:
    def __init__(self, n_lags=5, n_estimators=200, random_state=42):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def _make_features(self, y):
        X, Y = [], []
        for i in range(self.n_lags, len(y)):
            X.append(y[i - self.n_lags:i])
            Y.append(y[i])
        return np.array(X), np.array(Y)

    def fit(self, y):
        y = np.array(y, dtype=float)
        self._history = y.copy()
        n_lags = min(self.n_lags, len(y) // 3)
        self.n_lags = max(1, n_lags)
        X, Y = self._make_features(y)
        if len(X) < 5:
            self.n_lags = 1
            X, Y = self._make_features(y)
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        )
        self.model.fit(X_scaled, Y)
        return self

    def predict(self, h):
        buf = list(self._history)
        forecasts = []
        for _ in range(h):
            x = np.array(buf[-self.n_lags:]).reshape(1, -1)
            x_scaled = self.scaler.transform(x)
            pred = self.model.predict(x_scaled)[0]
            forecasts.append(pred)
            buf.append(pred)
        return np.array(forecasts)


# ─────────────────────────────────────────────────────────────
# 5. Gradient Boosting Forecaster
# ─────────────────────────────────────────────────────────────

class GBForecaster:
    def __init__(self, n_lags=5, n_estimators=200, random_state=42):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()

    def _make_features(self, y):
        X, Y = [], []
        for i in range(self.n_lags, len(y)):
            X.append(y[i - self.n_lags:i])
            Y.append(y[i])
        return np.array(X), np.array(Y)

    def fit(self, y):
        y = np.array(y, dtype=float)
        self._history = y.copy()
        self.n_lags = max(1, min(self.n_lags, len(y) // 3))
        X, Y = self._make_features(y)
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state,
            learning_rate=0.05, max_depth=3
        )
        self.model.fit(X_scaled, Y)
        return self

    def predict(self, h):
        buf = list(self._history)
        forecasts = []
        for _ in range(h):
            x = np.array(buf[-self.n_lags:]).reshape(1, -1)
            x_scaled = self.scaler.transform(x)
            pred = self.model.predict(x_scaled)[0]
            forecasts.append(pred)
            buf.append(pred)
        return np.array(forecasts)


# ─────────────────────────────────────────────────────────────
# 6. SVR Forecaster
# ─────────────────────────────────────────────────────────────

class SVRForecaster:
    def __init__(self, n_lags=4, C=100, epsilon=0.1, kernel="rbf"):
        self.n_lags = n_lags
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _make_features(self, y):
        X, Y = [], []
        for i in range(self.n_lags, len(y)):
            X.append(y[i - self.n_lags:i])
            Y.append(y[i])
        return np.array(X), np.array(Y)

    def fit(self, y):
        y = np.array(y, dtype=float)
        self._history = y.copy()
        self.n_lags = max(1, min(self.n_lags, len(y) // 3))
        X, Y = self._make_features(y)
        X_s = self.scaler_X.fit_transform(X)
        Y_s = self.scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()
        self.model = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        self.model.fit(X_s, Y_s)
        return self

    def predict(self, h):
        buf = list(self._history)
        forecasts = []
        for _ in range(h):
            x = np.array(buf[-self.n_lags:]).reshape(1, -1)
            x_s = self.scaler_X.transform(x)
            pred_s = self.model.predict(x_s)
            pred = self.scaler_y.inverse_transform(pred_s.reshape(-1, 1))[0, 0]
            forecasts.append(pred)
            buf.append(pred)
        return np.array(forecasts)


# ─────────────────────────────────────────────────────────────
# 7. Simple Ensemble (weighted average of best N models)
# ─────────────────────────────────────────────────────────────

class EnsembleForecaster:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1 / len(models)] * len(models)

    def fit(self, y):
        for m in self.models:
            m.fit(y)
        return self

    def predict(self, h):
        preds = np.array([m.predict(h) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)


# ─────────────────────────────────────────────────────────────
# Adaptive Model Selector via Walk-Forward Validation
# ─────────────────────────────────────────────────────────────

def adaptive_model_selection(y, h_cv=3, verbose=True):
    """
    Evaluate candidate models via walk-forward cross-validation.
    Returns sorted DataFrame of results + best model instance fitted on full data.
    """
    y = np.array(y, dtype=float)
    n = len(y)

    # Decide candidate set based on series length
    if n < 15:
        label = "SHORT (<15 yrs) → AR-based models preferred"
        candidates = {
            "HoltWinters": HoltWinters(),
            "ARIMALite(2,1)": ARIMALite(p=2, d=1),
            "PolyTrend(2)": PolynomialTrend(degree=2),
            "PolyTrend(3)": PolynomialTrend(degree=3),
        }
    elif n < 30:
        label = "MEDIUM (15-30 yrs) → ML + statistical models"
        candidates = {
            "HoltWinters": HoltWinters(),
            "ARIMALite(2,1)": ARIMALite(p=2, d=1),
            "RandomForest": RFForecaster(n_lags=4),
            "GradientBoosting": GBForecaster(n_lags=4),
            "SVR(rbf)": SVRForecaster(n_lags=4),
            "PolyTrend(2)": PolynomialTrend(degree=2),
        }
    else:
        label = "LONG (>30 yrs) → Advanced ML / ensemble models preferred"
        candidates = {
            "HoltWinters": HoltWinters(),
            "ARIMALite(3,1)": ARIMALite(p=3, d=1),
            "RandomForest": RFForecaster(n_lags=6),
            "GradientBoosting": GBForecaster(n_lags=6),
            "SVR(rbf)": SVRForecaster(n_lags=5),
            "PolyTrend(2)": PolynomialTrend(degree=2),
        }

    if verbose:
        print(f"\n  Series length: {n} years — {label}")
        print(f"  Testing {len(candidates)} models...")

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, n // 5)))
    results = []

    for name, model in candidates.items():
        cv_rmse, cv_mape, cv_r2 = [], [], []
        for train_idx, test_idx in tscv.split(y):
            if len(train_idx) < 5:
                continue
            try:
                m_clone = type(model)(**{k: v for k, v in model.__dict__.items()
                                        if not k.startswith("_")
                                        and k not in ["coeffs", "intercept", "model", "scaler",
                                                       "scaler_X", "scaler_y", "_fitted",
                                                       "_level", "_trend", "_diff",
                                                       "_last_p", "_history", "_poly",
                                                       "model_fit", "_start_year"]})
                m_clone.fit(y[train_idx])
                h = len(test_idx)
                preds = m_clone.predict(h)
                actuals = y[test_idx]
                min_len = min(len(preds), len(actuals))
                if min_len < 1:
                    continue
                cv_rmse.append(rmse(actuals[:min_len], preds[:min_len]))
                cv_mape.append(mape(actuals[:min_len], preds[:min_len]))
                cv_r2.append(r2(actuals[:min_len], preds[:min_len]))
            except Exception as e:
                if verbose:
                    print(f"    {name}: Failed ({str(e)[:50]})")
                continue

        if cv_rmse:
            results.append({
                "Model": name,
                "CV-RMSE": round(np.mean(cv_rmse), 4),
                "CV-MAPE (%)": round(np.mean(cv_mape), 2),
                "CV-R²": round(np.mean(cv_r2), 4),
                "Std-RMSE": round(np.std(cv_rmse), 4),
            })

    if not results:
        results.append({"Model": "PolyTrend(2)", "CV-RMSE": 0, "CV-MAPE (%)": 0, "CV-R²": 0, "Std-RMSE": 0})

    df = pd.DataFrame(results).sort_values("CV-RMSE")

    # Fit best model on full series
    best_name = df.iloc[0]["Model"]
    best_model = candidates.get(best_name, PolynomialTrend(2))
    best_model.fit(y)

    if verbose:
        print(f"  ✓ Best model: {best_name} (CV-RMSE={df.iloc[0]['CV-RMSE']:.4f}, "
              f"CV-MAPE={df.iloc[0]['CV-MAPE (%)']:.2f}%, CV-R²={df.iloc[0]['CV-R²']:.4f})")

    return df, best_model, best_name, label
