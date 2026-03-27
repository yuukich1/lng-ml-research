import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class AnomalyModel:
    """Rank vessels by how unusual their transit pattern looks."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.feature_columns = [
            "Zone",
            "Status",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "duration_seconds",
            "duration_hours",
            "duration_log",
            "vessel_name_len",
            "zone_duration_median",
            "zone_duration_mean",
            "zone_sample_count",
            "duration_vs_zone_median",
            "duration_delta_from_zone_mean",
            "is_completed",
        ]
        self.numeric_features = [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "duration_seconds",
            "duration_hours",
            "duration_log",
            "vessel_name_len",
            "zone_duration_median",
            "zone_duration_mean",
            "zone_sample_count",
            "duration_vs_zone_median",
            "duration_delta_from_zone_mean",
            "is_completed",
        ]
        self.categorical_features = ["Zone", "Status"]
        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", StandardScaler(), self.numeric_features),
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore"),
                                self.categorical_features,
                            ),
                        ]
                    ),
                ),
                (
                    "model",
                    IsolationForest(
                        contamination=self.contamination,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        model_input = df[self.feature_columns].copy()
        result = df.copy()

        result["anomaly_label"] = self.pipeline.fit_predict(model_input)
        result["is_anomaly"] = result["anomaly_label"] == -1

        raw_iforest_score = -self.pipeline.named_steps["model"].score_samples(
            self.pipeline.named_steps["preprocessor"].transform(model_input)
        )
        result["iforest_score"] = raw_iforest_score

        zone_ratio = result["duration_vs_zone_median"].clip(lower=0)
        zone_ratio_scaled = zone_ratio / max(zone_ratio.max(), 1)

        # Blend model output with an interpretable "longer than zone typical" signal.
        result["risk_score"] = (0.7 * self._minmax(raw_iforest_score)) + (0.3 * zone_ratio_scaled)
        result["risk_rank"] = result["risk_score"].rank(method="dense", ascending=False).astype(int)

        return result.sort_values(["risk_score", "duration_seconds"], ascending=[False, False])

    def top_anomalies(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        scored = self.detect(df)
        columns = [
            "risk_rank",
            "risk_score",
            "is_anomaly",
            "Vessel",
            "Zone",
            "Status",
            "duration_seconds",
            "duration_hours",
            "duration_vs_zone_median",
            "Entry Datetime",
            "Exit Datetime",
        ]
        existing_columns = [column for column in columns if column in scored.columns]
        return scored.loc[:, existing_columns].head(top_n)

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        value_range = values.max() - values.min()
        if value_range == 0:
            return np.zeros_like(values, dtype=float)
        return (values - values.min()) / value_range
