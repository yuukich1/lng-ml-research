from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd


@dataclass
class STSConfig:
    min_overlap_minutes: int = 30
    min_overlap_ratio: float = 0.3
    top_n: int = 20


class STSAnalyzer:
    """Heuristic STS candidate detection from shared zone/time overlap."""

    def __init__(self, config: STSConfig | None = None):
        self.config = config or STSConfig()

    def build_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        intervals = self._normalize_interval_input(df)
        intervals["Entry Datetime"] = pd.to_datetime(intervals["Entry Datetime"], utc=True, errors="coerce")
        intervals["Exit Datetime"] = pd.to_datetime(intervals["Exit Datetime"], utc=True, errors="coerce")
        intervals["Duration Seconds"] = pd.to_numeric(intervals["Duration Seconds"], errors="coerce").fillna(0)

        inferred_exit = intervals["Entry Datetime"] + pd.to_timedelta(
            intervals["Duration Seconds"], unit="s"
        )
        intervals["effective_exit"] = intervals["Exit Datetime"].fillna(inferred_exit)
        intervals["effective_end_source"] = intervals["Exit Datetime"].notna().map(
            {True: "observed_exit", False: "entry_plus_duration"}
        )

        return intervals.dropna(subset=["Entry Datetime", "effective_exit", "Zone", "Vessel"])

    def find_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        if self._is_precomputed_candidate_input(df):
            return self._normalize_precomputed_candidates(df)

        intervals = self.build_intervals(df)
        if intervals.empty:
            return pd.DataFrame()

        candidates: list[dict] = []

        for zone, zone_df in intervals.groupby("Zone", dropna=False):
            records = zone_df.sort_values("Entry Datetime").to_dict("records")
            for left, right in combinations(records, 2):
                overlap_seconds = self._overlap_seconds(
                    left["Entry Datetime"],
                    left["effective_exit"],
                    right["Entry Datetime"],
                    right["effective_exit"],
                )
                if overlap_seconds <= 0:
                    continue

                left_duration = max(float(left["Duration Seconds"]), 1.0)
                right_duration = max(float(right["Duration Seconds"]), 1.0)
                min_duration = min(left_duration, right_duration)
                overlap_ratio = overlap_seconds / min_duration

                if overlap_seconds < self.config.min_overlap_minutes * 60:
                    continue
                if overlap_ratio < self.config.min_overlap_ratio:
                    continue

                entry_delta_minutes = abs(
                    (left["Entry Datetime"] - right["Entry Datetime"]).total_seconds()
                ) / 60.0
                duration_similarity = min(left_duration, right_duration) / max(
                    left_duration, right_duration
                )

                sts_score = (
                    0.5 * min(overlap_ratio, 1.0)
                    + 0.3 * duration_similarity
                    + 0.2 * (1 / (1 + entry_delta_minutes / 60.0))
                )

                candidates.append(
                    {
                        "Zone": zone,
                        "Vessel A": left["Vessel"],
                        "Vessel B": right["Vessel"],
                        "MMSI A": left["MMSI"],
                        "MMSI B": right["MMSI"],
                        "Status A": left["Status"],
                        "Status B": right["Status"],
                        "Entry A": left["Entry Datetime"],
                        "Entry B": right["Entry Datetime"],
                        "Effective Exit A": left["effective_exit"],
                        "Effective Exit B": right["effective_exit"],
                        "Overlap Hours": overlap_seconds / 3600.0,
                        "Overlap Ratio": overlap_ratio,
                        "Entry Delta Minutes": entry_delta_minutes,
                        "Duration Similarity": duration_similarity,
                        "STS Score": sts_score,
                        "End Source A": left["effective_end_source"],
                        "End Source B": right["effective_end_source"],
                    }
                )

        if not candidates:
            return pd.DataFrame(
                columns=[
                    "Zone",
                    "Vessel A",
                    "Vessel B",
                    "Overlap Hours",
                    "Overlap Ratio",
                    "Entry Delta Minutes",
                    "Duration Similarity",
                    "STS Score",
                ]
            )

        result = pd.DataFrame(candidates).sort_values(
            ["STS Score", "Overlap Hours", "Overlap Ratio"],
            ascending=[False, False, False],
        )
        return result.head(self.config.top_n).reset_index(drop=True)

    @staticmethod
    def _normalize_interval_input(df: pd.DataFrame) -> pd.DataFrame:
        if {"Entry Datetime", "Vessel", "Zone"}.issubset(df.columns):
            intervals = df.copy()
        elif {"entry_datetime", "name", "zone"}.issubset(df.columns):
            intervals = pd.DataFrame(
                {
                    "Entry Datetime": df["entry_datetime"],
                    "Exit Datetime": df.get("exit_datetime"),
                    "Duration Seconds": df.get("duration_seconds"),
                    "Zone": df["zone"],
                    "Vessel": df["name"],
                    "MMSI": df.get("mmsi"),
                    "Status": df.get("status", "unknown"),
                }
            )
        else:
            missing = ["Entry Datetime/Vessel/Zone", "entry_datetime/name/zone"]
            raise KeyError(f"Unsupported STS input format. Expected one of: {missing}")

        for column in ["MMSI", "Status"]:
            if column not in intervals.columns:
                intervals[column] = pd.NA
        if "Duration Seconds" not in intervals.columns:
            intervals["Duration Seconds"] = pd.NA
        if "Exit Datetime" not in intervals.columns:
            intervals["Exit Datetime"] = pd.NaT

        missing_duration = intervals["Duration Seconds"].isna()
        inferred_duration = (
            pd.to_datetime(intervals["Exit Datetime"], utc=True, errors="coerce")
            - pd.to_datetime(intervals["Entry Datetime"], utc=True, errors="coerce")
        ).dt.total_seconds()
        intervals.loc[missing_duration, "Duration Seconds"] = inferred_duration.loc[missing_duration]

        return intervals

    def _normalize_precomputed_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "Zone": df.get("zone"),
                "Vessel A": df.get("vessel_a_name"),
                "Vessel B": df.get("vessel_b_name"),
                "MMSI A": df.get("vessel_a_mmsi"),
                "MMSI B": df.get("vessel_b_mmsi"),
                "Status A": pd.NA,
                "Status B": pd.NA,
                "Entry A": pd.NaT,
                "Entry B": pd.NaT,
                "Effective Exit A": pd.NaT,
                "Effective Exit B": pd.NaT,
                "Overlap Hours": pd.to_numeric(df.get("overlap_hours"), errors="coerce"),
                "Overlap Ratio": pd.NA,
                "Entry Delta Minutes": pd.NA,
                "Duration Similarity": pd.NA,
                "STS Score": pd.to_numeric(df.get("sts_score"), errors="coerce"),
                "End Source A": "precomputed_candidate",
                "End Source B": "precomputed_candidate",
            }
        )
        result = result.sort_values(
            ["STS Score", "Overlap Hours"],
            ascending=[False, False],
        )
        return result.head(self.config.top_n).reset_index(drop=True)

    @staticmethod
    def _is_precomputed_candidate_input(df: pd.DataFrame) -> bool:
        return {
            "vessel_a_mmsi",
            "vessel_b_mmsi",
            "vessel_a_name",
            "vessel_b_name",
            "sts_score",
        }.issubset(df.columns)

    @staticmethod
    def _overlap_seconds(
        start_a: pd.Timestamp,
        end_a: pd.Timestamp,
        start_b: pd.Timestamp,
        end_b: pd.Timestamp,
    ) -> float:
        latest_start = max(start_a, start_b)
        earliest_end = min(end_a, end_b)
        return max((earliest_end - latest_start).total_seconds(), 0.0)
