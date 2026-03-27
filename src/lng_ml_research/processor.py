import json
from pathlib import Path

import numpy as np
import pandas as pd


class LNGDataProcessor:
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Load interval-like training data from JSON and JSONL sources."""
        event_frames = self._load_event_frames()
        frames = [frame for frame in event_frames if not frame.empty]

        if not frames:
            print("Supported training datasets were not found in data/raw")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df.columns = df.columns.str.strip()

        if "Entry Datetime" in df.columns:
            df["Entry Datetime"] = pd.to_datetime(df["Entry Datetime"], utc=True, errors="coerce")
        if "Exit Datetime" in df.columns:
            df["Exit Datetime"] = pd.to_datetime(df["Exit Datetime"], utc=True, errors="coerce")
        if "Duration Seconds" in df.columns:
            df["Duration Seconds"] = pd.to_numeric(df["Duration Seconds"], errors="coerce")

        if "Duration HMS" not in df.columns:
            df["Duration HMS"] = pd.to_timedelta(
                df["Duration Seconds"].fillna(0), unit="s"
            ).astype("string")

        dedupe_keys = [
            column
            for column in ["MMSI", "Vessel", "Zone", "Entry Datetime", "Exit Datetime", "Status"]
            if column in df.columns
        ]
        if dedupe_keys:
            df = df.drop_duplicates(subset=dedupe_keys, keep="first")

        return df.reset_index(drop=True)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build numeric and categorical features for anomaly detection."""
        if df.empty:
            return df

        features_df = df.copy()
        features_df.columns = features_df.columns.str.strip()

        features_df["Status"] = features_df["Status"].fillna("unknown")
        features_df["Zone"] = features_df["Zone"].fillna("unknown")
        features_df["Vessel"] = features_df["Vessel"].fillna("unknown")

        features_df["hour"] = features_df["Entry Datetime"].dt.hour.fillna(-1).astype(int)
        features_df["day_of_week"] = features_df["Entry Datetime"].dt.dayofweek.fillna(-1).astype(int)
        features_df["month"] = features_df["Entry Datetime"].dt.month.fillna(-1).astype(int)
        features_df["is_weekend"] = features_df["day_of_week"].isin([5, 6]).astype(int)

        features_df["duration_seconds"] = features_df["Duration Seconds"].fillna(0)
        features_df["duration_hours"] = features_df["duration_seconds"] / 3600.0
        features_df["duration_log"] = np.log1p(features_df["duration_seconds"].clip(lower=0))
        features_df["is_completed"] = (features_df["Status"] == "completed").astype(int)
        features_df["vessel_name_len"] = features_df["Vessel"].str.len()

        zone_median = (
            features_df.groupby("Zone")["duration_seconds"]
            .transform("median")
            .fillna(features_df["duration_seconds"].median())
        )
        zone_mean = (
            features_df.groupby("Zone")["duration_seconds"]
            .transform("mean")
            .fillna(features_df["duration_seconds"].mean())
        )
        zone_count = features_df.groupby("Zone")["Zone"].transform("count")

        features_df["zone_duration_median"] = zone_median
        features_df["zone_duration_mean"] = zone_mean
        features_df["zone_sample_count"] = zone_count
        features_df["duration_vs_zone_median"] = features_df["duration_seconds"] / zone_median.clip(lower=1)
        features_df["duration_delta_from_zone_mean"] = features_df["duration_seconds"] - zone_mean

        return features_df

    def _load_event_frames(self) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for file_path in Path(self.data_path).rglob("*"):
            if file_path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            records = self._extract_event_records(file_path)
            if not records:
                continue
            frame = self._normalize_event_records(records)
            if not frame.empty:
                frames.append(frame)
        return frames

    def _extract_event_records(self, file_path: Path) -> list[dict]:
        if file_path.suffix.lower() == ".jsonl":
            records = []
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if self._is_event_record(payload):
                        records.append(payload)
            return records

        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            return [item for item in payload if self._is_event_record(item)]

        if isinstance(payload, dict):
            if "vessel_zone_events" in payload and isinstance(payload["vessel_zone_events"], list):
                return [item for item in payload["vessel_zone_events"] if self._is_event_record(item)]
            if "data" in payload and isinstance(payload["data"], list):
                return [item for item in payload["data"] if self._is_event_record(item)]
            if self._is_event_record(payload):
                return [payload]

        return []

    @staticmethod
    def _is_event_record(payload: object) -> bool:
        return isinstance(payload, dict) and {
            "mmsi",
            "name",
            "zone",
            "entry_datetime",
        }.issubset(payload.keys())

    @staticmethod
    def _normalize_event_records(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if df.empty:
            return df

        normalized = pd.DataFrame(
            {
                "MMSI": pd.to_numeric(df.get("mmsi"), errors="coerce"),
                "Vessel": df.get("name"),
                "Zone": df.get("zone"),
                "Entry Datetime": pd.to_datetime(df.get("entry_datetime"), utc=True, errors="coerce"),
                "Exit Datetime": pd.to_datetime(df.get("exit_datetime"), utc=True, errors="coerce"),
                "Duration Seconds": pd.to_numeric(df.get("duration_seconds"), errors="coerce"),
                "Status": df.get("status"),
            }
        )

        missing_duration = normalized["Duration Seconds"].isna()
        normalized.loc[missing_duration, "Duration Seconds"] = (
            normalized.loc[missing_duration, "Exit Datetime"]
            - normalized.loc[missing_duration, "Entry Datetime"]
        ).dt.total_seconds()
        normalized["Duration HMS"] = pd.to_timedelta(
            normalized["Duration Seconds"].fillna(0), unit="s"
        ).astype("string")

        return normalized.dropna(subset=["Entry Datetime", "Vessel", "Zone"]).reset_index(drop=True)
