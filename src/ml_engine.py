"""
ML Engine: Random Forest classifier that trains on ETH historical candles
and predicts the direction of the next candle.

Features used (20+):
- RSI, MACD, MACD histogram, Stochastic K/D
- EMA alignment score, EMA slopes (9, 21, 50)
- Bollinger Band %B, BB width (squeeze)
- ATR normalized by price
- Rate of change (5, 10 periods)
- Volume proxy: candle body size / range
- Candle direction (last 3 candles)
- RSI trend (rising/falling)
- Price vs EMA 50, 200

Target: 1 = next close > current close (bullish), 0 = bearish
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from dataclasses import dataclass
from typing import List


@dataclass
class MLPrediction:
    direction: str          # BULLISH / BEARISH
    probability: float      # 0-100 confidence
    model_accuracy: float   # cross-val accuracy on training data
    feature_importances: List[dict]  # top features driving the prediction
    signal_strength: str    # STRONG / MODERATE / WEAK


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from indicator DataFrame."""
    f = pd.DataFrame(index=df.index)

    close = df["close"]

    # Momentum
    f["rsi"]         = df["rsi"]
    f["rsi_slope"]   = df["rsi"].diff(3)
    f["macd"]        = df["macd"]
    f["macd_hist"]   = df["macd_hist"]
    f["macd_hist_slope"] = df["macd_hist"].diff(2)
    f["stoch_k"]     = df["stoch_k"]
    f["stoch_d"]     = df["stoch_d"]
    f["stoch_diff"]  = df["stoch_k"] - df["stoch_d"]

    # Trend structure
    f["ema_align"]   = df["ema_align"]
    f["ema9_slope"]  = df["ema_9"].pct_change(3) * 100
    f["ema21_slope"] = df["ema_21"].pct_change(3) * 100
    f["ema50_slope"] = df["ema_50"].pct_change(3) * 100
    f["price_vs_ema50"]  = (close - df["ema_50"])  / df["ema_50"]  * 100
    f["price_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"] * 100

    # Volatility
    f["bb_pct"]      = df["bb_pct"]
    f["bb_width"]    = df["bb_width"]
    f["atr_pct"]     = df["atr"] / close * 100   # ATR as % of price

    # Momentum
    f["roc_5"]       = df["roc_10"].shift(5)     # 5-period ROC
    f["roc_10"]      = df["roc_10"]

    # Candle structure
    body  = (close - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    f["body_ratio"]  = body / range_              # 0=doji, 1=full candle
    f["candle_dir"]  = np.sign(close - df["open"])
    f["candle_dir1"] = f["candle_dir"].shift(1)
    f["candle_dir2"] = f["candle_dir"].shift(2)

    # RSI regime
    f["rsi_ob"]      = (df["rsi"] > 70).astype(float)
    f["rsi_os"]      = (df["rsi"] < 30).astype(float)

    return f


FEATURE_NAMES = [
    "rsi", "rsi_slope", "macd", "macd_hist", "macd_hist_slope",
    "stoch_k", "stoch_d", "stoch_diff",
    "ema_align", "ema9_slope", "ema21_slope", "ema50_slope",
    "price_vs_ema50", "price_vs_ema200",
    "bb_pct", "bb_width", "atr_pct",
    "roc_5", "roc_10",
    "body_ratio", "candle_dir", "candle_dir1", "candle_dir2",
    "rsi_ob", "rsi_os",
]


def train_and_predict(df: pd.DataFrame) -> MLPrediction:
    """
    Train a Random Forest + Gradient Boosting ensemble on historical ETH data,
    then predict the next candle direction.
    """
    features = _build_features(df)

    # Target: 1 if next candle closes higher
    target = (df["close"].shift(-1) > df["close"]).astype(int)

    # Align and drop NaNs
    data = features[FEATURE_NAMES].copy()
    data["target"] = target
    data = data.dropna()

    if len(data) < 15:
        return MLPrediction(
            direction="NEUTRAL",
            probability=50.0,
            model_accuracy=0.0,
            feature_importances=[],
            signal_strength="WEAK",
        )

    X = data[FEATURE_NAMES].values
    y = data["target"].values

    # Don't use the last row for training (that's what we predict)
    X_train, y_train = X[:-1], y[:-1]
    X_pred = X[-1:].copy()

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled  = scaler.transform(X_pred)

    # Ensemble: Random Forest + Gradient Boosting
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )

    # Cross-validation accuracy (use min 3 folds, or fewer if data is small)
    n_folds = min(5, len(X_train) // 3)
    if n_folds >= 2:
        try:
            cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=n_folds, scoring="accuracy")
            model_accuracy = round(float(cv_scores.mean()) * 100, 1)
        except Exception:
            model_accuracy = 0.0
    else:
        model_accuracy = 0.0

    # Train both models
    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)

    # Ensemble probability (average)
    rf_proba = rf.predict_proba(X_pred_scaled)[0]
    gb_proba = gb.predict_proba(X_pred_scaled)[0]
    avg_proba = (rf_proba + gb_proba) / 2

    bull_prob  = float(avg_proba[1]) * 100
    bear_prob  = float(avg_proba[0]) * 100
    direction  = "BULLISH" if bull_prob >= 50 else "BEARISH"
    probability = bull_prob if direction == "BULLISH" else bear_prob

    # Signal strength based on confidence
    if probability >= 70:
        signal_strength = "STRONG"
    elif probability >= 58:
        signal_strength = "MODERATE"
    else:
        signal_strength = "WEAK"

    # Top feature importances from RF
    importances = rf.feature_importances_
    fi = sorted(
        [{"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]) * 100, 1)}
         for i in range(len(FEATURE_NAMES))],
        key=lambda x: x["importance"], reverse=True
    )[:6]

    return MLPrediction(
        direction=direction,
        probability=round(probability, 1),
        model_accuracy=model_accuracy,
        feature_importances=fi,
        signal_strength=signal_strength,
    )
