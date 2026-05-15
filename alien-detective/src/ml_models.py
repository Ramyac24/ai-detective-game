"""
ml_models.py
ML classification and clustering for the alien signal mystery game.

Models
------
- SignalClassifier  : RandomForest multi-class classifier (signal type detection)
- SignalClusterer   : KMeans + optional DBSCAN for origin-sector mapping
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from src.data_generator import get_feature_matrix, FEATURE_COLS

MODEL_DIR = "models"


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class SignalClassifier:
    """
    Trains a RandomForest to classify alien signals into:
    Beacon | Warning | DataTransmission | Anomaly | Interference
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(MODEL_DIR, "classifier.joblib")
        self.scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        self.encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

        self.model   = None
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()
        self.feature_cols = None
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the classifier on the dataset.
        Returns a metrics dict with accuracy, f1, confusion matrix.
        """
        X, y, feature_cols = get_feature_matrix(df)
        self.feature_cols = feature_cols

        y_enc = self.encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc
        )

        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        cm   = confusion_matrix(y_test, y_pred)
        labels = self.encoder.classes_.tolist()
        report = classification_report(y_test, y_pred,
                                       target_names=labels, output_dict=True)

        # Cross-val
        cv_scores = cross_val_score(self.model, X_scaled, y_enc, cv=5,
                                    scoring="accuracy", n_jobs=-1)

        self.is_trained = True
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model,   self.model_path)
        joblib.dump(self.scaler,  self.scaler_path)
        joblib.dump(self.encoder, self.encoder_path)

        return {
            "accuracy":        round(acc, 4),
            "f1_weighted":     round(f1, 4),
            "cv_mean":         round(cv_scores.mean(), 4),
            "cv_std":          round(cv_scores.std(), 4),
            "confusion_matrix": cm.tolist(),
            "labels":          labels,
            "report":          report,
            "feature_importance": dict(zip(
                feature_cols,
                self.model.feature_importances_.tolist()
            )),
        }

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict signal classes for a DataFrame. Returns df with new columns."""
        if not self.is_trained:
            self.load()
        X, _, _ = get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        y_pred   = self.model.predict(X_scaled)
        proba    = self.model.predict_proba(X_scaled)

        result = df.copy()
        result["predicted_class"]      = self.encoder.inverse_transform(y_pred)
        result["confidence"]           = proba.max(axis=1).round(4)
        result["anomaly_probability"]  = proba[
            :, list(self.encoder.classes_).index("Anomaly")
        ].round(4)
        return result

    def predict_single(self, features: list) -> dict:
        """
        Predict a single signal.
        features: [frequency_mhz, amplitude_db, duration_sec,
                   pulse_rate_hz, noise_ratio, sector_x, sector_y, modulation_enc]
        """
        if not self.is_trained:
            self.load()
        arr = np.array(features).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)
        pred  = self.model.predict(arr_scaled)[0]
        proba = self.model.predict_proba(arr_scaled)[0]
        label = self.encoder.inverse_transform([pred])[0]
        return {
            "predicted_class": label,
            "probabilities": dict(zip(self.encoder.classes_, proba.round(4).tolist()))
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def load(self):
        """Load pre-trained model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No trained model found at {self.model_path}. "
                "Please run train() first."
            )
        self.model   = joblib.load(self.model_path)
        self.scaler  = joblib.load(self.scaler_path)
        self.encoder = joblib.load(self.encoder_path)
        self.is_trained = True

    def is_ready(self) -> bool:
        return os.path.exists(self.model_path)


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL CLUSTERER
# ══════════════════════════════════════════════════════════════════════════════

class SignalClusterer:
    """
    Clusters signals by (sector_x, sector_y) to reveal hidden origin patterns.
    Uses KMeans for general mapping, DBSCAN to isolate anomaly hotspots.
    """

    def __init__(self):
        self.kmeans = None
        self.dbscan = None
        self.pca    = PCA(n_components=2)
        self.scaler = StandardScaler()

    # ── KMeans ────────────────────────────────────────────────────────────────
    def fit_kmeans(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Cluster all signals using KMeans.
        Returns df with 'kmeans_cluster' column.
        """
        coords = df[["sector_x", "sector_y", "frequency_mhz",
                     "amplitude_db", "noise_ratio"]].values
        coords_scaled = self.scaler.fit_transform(coords)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42,
                             n_init="auto", max_iter=300)
        labels = self.kmeans.fit_predict(coords_scaled)

        result = df.copy()
        result["kmeans_cluster"] = labels

        # Compute cluster centers in original space
        centers_scaled = self.kmeans.cluster_centers_
        # Only invert first 2 dims (sector_x, sector_y)
        dummy = np.zeros((n_clusters, coords.shape[1]))
        dummy[:, :2] = centers_scaled[:, :2]
        centers_orig = self.scaler.inverse_transform(dummy)

        self.cluster_centers_ = centers_orig[:, :2]
        return result

    # ── DBSCAN (anomaly hotspot detection) ────────────────────────────────────
    def fit_dbscan(self, df: pd.DataFrame,
                   eps: float = 25.0,
                   min_samples: int = 3) -> pd.DataFrame:
        """
        DBSCAN on spatial coords to find dense anomaly clusters.
        Returns df with 'dbscan_cluster' column (-1 = noise/outlier).
        """
        coords = df[["sector_x", "sector_y"]].values
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan.fit_predict(coords)

        result = df.copy()
        result["dbscan_cluster"] = labels
        return result

    # ── PCA for 2D projection ─────────────────────────────────────────────────
    def pca_projection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Project high-dimensional features to 2D via PCA for visualisation.
        Returns df with 'pca_x', 'pca_y' columns.
        """
        from src.data_generator import get_feature_matrix
        X, _, _ = get_feature_matrix(df)
        X_std  = StandardScaler().fit_transform(X)
        proj   = self.pca.fit_transform(X_std)
        result = df.copy()
        result["pca_x"] = proj[:, 0]
        result["pca_y"] = proj[:, 1]
        result["explained_variance"] = self.pca.explained_variance_ratio_.sum()
        return result

    # ── Mystery score ─────────────────────────────────────────────────────────
    def anomaly_concentration_score(self, df: pd.DataFrame) -> float:
        """
        Returns a 0-1 score representing how tightly Anomaly signals cluster.
        High score (>0.7) = "something is out there".
        """
        anomalies = df[df["signal_class"] == "Anomaly"][["sector_x", "sector_y"]]
        if len(anomalies) < 3:
            return 0.0
        # Coefficient of variation (lower = tighter cluster)
        std  = anomalies.std().mean()
        mean = anomalies.mean().mean()
        cv   = std / (mean + 1e-9)
        score = float(np.clip(1 - cv / 2, 0, 1))
        return round(score, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: train & predict in one call
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(df: pd.DataFrame) -> dict:
    """
    Run complete ML pipeline: classify + cluster.
    Returns a dict with annotated DataFrames and metrics.
    """
    clf = SignalClassifier()
    metrics = clf.train(df)
    df_classified = clf.predict(df)

    clust = SignalClusterer()
    df_kmeans = clust.fit_kmeans(df_classified, n_clusters=5)
    df_final  = clust.fit_dbscan(df_kmeans)
    df_final  = clust.pca_projection(df_final)

    mystery_score = clust.anomaly_concentration_score(df_final)

    return {
        "dataframe":     df_final,
        "metrics":       metrics,
        "mystery_score": mystery_score,
        "classifier":    clf,
        "clusterer":     clust,
    }
