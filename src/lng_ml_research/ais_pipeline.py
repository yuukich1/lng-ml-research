from __future__ import annotations

import json
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


@dataclass
class ZoneConfig:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class AISDatasetBuilder:
    REQUIRED_COLUMNS = [
        "observed_at",
        "vessel_id",
        "name",
        "imo",
        "mmsi",
        "flag",
        "vessel_type",
        "deadweight",
        "latitude",
        "longitude",
        "speed_knots",
        "cog_degrees",
        "draught_meters",
        "nav_status",
        "destination",
        "position_source",
    ]

    OUTPUT_TABLES = {
        "observations": "ais_observations",
        "events": "vessel_zone_events",
        "sts": "sts_candidates",
        "loitering": "loitering_candidates",
        "congestion": "zone_congestion",
    }

    def __init__(self, event_gap_minutes: int = 360, distance_tolerance_minutes: int = 30):
        self.event_gap = pd.Timedelta(minutes=event_gap_minutes)
        self.distance_tolerance = pd.Timedelta(minutes=distance_tolerance_minutes)
        self.zones = [
            ZoneConfig("gibraltar", 35.5, 36.5, -6.0, -4.5),
            ZoneConfig("suez_canal", 29.5, 31.6, 32.2, 32.8),
            ZoneConfig("strait_of_malacca", 1.0, 7.0, 98.0, 104.5),
        ]

    def build_all(
        self, input_path: str | Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        observations = self.load_ais_data(input_path)
        events = self.build_zone_events(observations)
        sts_candidates = self.build_sts_candidates(observations, events)
        loitering_candidates = self.build_loitering_candidates(events)
        zone_congestion = self.build_zone_congestion(observations, events)
        summary = self.build_summary(
            observations,
            events,
            sts_candidates,
            loitering_candidates,
            zone_congestion,
        )
        return observations, events, sts_candidates, loitering_candidates, zone_congestion, summary

    def load_ais_data(self, input_path: str | Path) -> pd.DataFrame:
        records: list[dict] = []

        if self._is_url(str(input_path)):
            records.extend(self._load_records_from_url(str(input_path)))
        else:
            input_path = Path(input_path)
            files = self._discover_files(input_path)
            for file_path in files:
                records.extend(self._load_records_from_file(file_path))

        observations = pd.DataFrame(records)
        if observations.empty:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS + ["zone"])

        for column in self.REQUIRED_COLUMNS:
            if column not in observations.columns:
                observations[column] = None

        observations = observations[self.REQUIRED_COLUMNS].copy()
        observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True, errors="coerce")
        observations = observations.dropna(subset=["observed_at"]).copy()

        numeric_columns = [
            "vessel_id",
            "imo",
            "mmsi",
            "deadweight",
            "latitude",
            "longitude",
            "speed_knots",
            "cog_degrees",
            "draught_meters",
        ]
        for column in numeric_columns:
            observations[column] = pd.to_numeric(observations[column], errors="coerce")

        string_columns = [
            "name",
            "flag",
            "vessel_type",
            "nav_status",
            "destination",
            "position_source",
        ]
        for column in string_columns:
            observations[column] = (
                observations[column]
                .replace(r"^\s*$", pd.NA, regex=True)
                .where(observations[column].notna(), pd.NA)
            )

        observations["zone"] = observations.apply(
            lambda row: self._detect_zone(row["latitude"], row["longitude"]), axis=1
        )
        observations["date_utc"] = observations["observed_at"].dt.date.astype("string")
        observations["hour_utc"] = observations["observed_at"].dt.hour
        observations["is_destination_missing"] = observations["destination"].isna()
        observations["is_speed_missing"] = observations["speed_knots"].isna()
        observations["is_draught_missing"] = observations["draught_meters"].isna()

        observations = observations.drop_duplicates(
            subset=["mmsi", "observed_at", "latitude", "longitude"], keep="first"
        ).sort_values(["mmsi", "observed_at"])

        return observations.reset_index(drop=True)

    def build_zone_events(self, observations: pd.DataFrame) -> pd.DataFrame:
        if observations.empty:
            return pd.DataFrame()

        zone_obs = observations[observations["zone"].notna()].copy()
        if zone_obs.empty:
            return pd.DataFrame()

        zone_obs = zone_obs.sort_values(["mmsi", "observed_at"]).reset_index(drop=True)
        prev_mmsi = zone_obs["mmsi"].shift()
        prev_zone = zone_obs["zone"].shift()
        prev_time = zone_obs["observed_at"].shift()

        is_new_event = (
            zone_obs["mmsi"].ne(prev_mmsi)
            | zone_obs["zone"].ne(prev_zone)
            | ((zone_obs["observed_at"] - prev_time) > self.event_gap)
        )
        zone_obs["event_id"] = is_new_event.cumsum()

        agg = (
            zone_obs.groupby("event_id", dropna=False)
            .agg(
                mmsi=("mmsi", "first"),
                imo=("imo", "first"),
                name=("name", "first"),
                vessel_id=("vessel_id", "first"),
                flag=("flag", "first"),
                vessel_type=("vessel_type", "first"),
                deadweight=("deadweight", "first"),
                zone=("zone", "first"),
                entry_datetime=("observed_at", "min"),
                exit_datetime=("observed_at", "max"),
                observations_count=("observed_at", "size"),
                avg_speed_knots=("speed_knots", "mean"),
                min_speed_knots=("speed_knots", "min"),
                max_speed_knots=("speed_knots", "max"),
                avg_draught_meters=("draught_meters", "mean"),
                min_draught_meters=("draught_meters", "min"),
                max_draught_meters=("draught_meters", "max"),
                avg_cog_degrees=("cog_degrees", "mean"),
                start_latitude=("latitude", "first"),
                start_longitude=("longitude", "first"),
                end_latitude=("latitude", "last"),
                end_longitude=("longitude", "last"),
                centroid_latitude=("latitude", "mean"),
                centroid_longitude=("longitude", "mean"),
                position_source=("position_source", "first"),
            )
            .reset_index()
        )

        agg["duration_seconds"] = (
            agg["exit_datetime"] - agg["entry_datetime"]
        ).dt.total_seconds()
        agg["duration_hours"] = agg["duration_seconds"] / 3600.0
        agg["draught_change_meters"] = agg["max_draught_meters"] - agg["min_draught_meters"]

        next_obs = (
            observations.sort_values(["mmsi", "observed_at"])
            .groupby("mmsi", dropna=False)["observed_at"]
            .max()
            .rename("last_observed_at")
        )
        agg = agg.merge(next_obs, left_on="mmsi", right_index=True, how="left")
        agg["status"] = agg.apply(
            lambda row: "active" if row["exit_datetime"] == row["last_observed_at"] else "completed",
            axis=1,
        )
        agg = agg.drop(columns=["last_observed_at"])

        return agg

    def build_sts_candidates(self, observations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if observations.empty or events.empty:
            return pd.DataFrame()

        working_events = events.copy()
        working_events["entry_datetime_ts"] = pd.to_datetime(working_events["entry_datetime"], utc=True)
        working_events["exit_datetime_ts"] = pd.to_datetime(working_events["exit_datetime"], utc=True)

        candidates: list[dict] = []

        for zone, zone_events in working_events.groupby("zone", dropna=False):
            zone_events = zone_events.sort_values("entry_datetime_ts").reset_index(drop=True)
            for left_index in range(len(zone_events)):
                left = zone_events.iloc[left_index]
                for right_index in range(left_index + 1, len(zone_events)):
                    right = zone_events.iloc[right_index]
                    if left["mmsi"] == right["mmsi"]:
                        continue

                    overlap_start = max(left["entry_datetime_ts"], right["entry_datetime_ts"])
                    overlap_end = min(left["exit_datetime_ts"], right["exit_datetime_ts"])
                    overlap_seconds = self._overlap_seconds(
                        left["entry_datetime_ts"],
                        left["exit_datetime_ts"],
                        right["entry_datetime_ts"],
                        right["exit_datetime_ts"],
                    )
                    if overlap_seconds <= 0:
                        continue

                    overlap_hours = overlap_seconds / 3600.0
                    min_event_duration = max(min(left["duration_seconds"], right["duration_seconds"]), 1.0)
                    overlap_ratio = overlap_seconds / min_event_duration

                    distance_stats = self._pair_distance_stats(
                        observations=observations,
                        zone=zone,
                        vessel_a_mmsi=left["mmsi"],
                        vessel_b_mmsi=right["mmsi"],
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                    )
                    avg_distance_nm = distance_stats["avg_distance_nm"]
                    min_distance_nm = distance_stats["min_distance_nm"]

                    distance_score = 0.0
                    if pd.notna(min_distance_nm):
                        distance_score = max(0.0, 1.0 - min(min_distance_nm / 5.0, 1.0))

                    mean_speed = pd.Series(
                        [left["avg_speed_knots"], right["avg_speed_knots"]], dtype="float64"
                    ).mean()
                    if pd.isna(mean_speed):
                        mean_speed = 0.0

                    draught_change_total = sum(
                        max(float(value), 0.0)
                        for value in [left["draught_change_meters"], right["draught_change_meters"]]
                        if pd.notna(value)
                    )

                    speed_score = 1.0 - min(
                        max(mean_speed, 0.0) / 5.0,
                        1.0,
                    )
                    draught_score = min(draught_change_total, 4.0) / 4.0

                    sts_score = (
                        0.35 * min(overlap_ratio, 1.0)
                        + 0.30 * distance_score
                        + 0.20 * speed_score
                        + 0.15 * draught_score
                    )

                    candidates.append(
                        {
                            "vessel_a_mmsi": left["mmsi"],
                            "vessel_b_mmsi": right["mmsi"],
                            "vessel_a_name": left["name"],
                            "vessel_b_name": right["name"],
                            "zone": zone,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_seconds": overlap_seconds,
                            "overlap_hours": overlap_hours,
                            "avg_distance_nm": avg_distance_nm,
                            "min_distance_nm": min_distance_nm,
                            "vessel_a_avg_speed": left["avg_speed_knots"],
                            "vessel_b_avg_speed": right["avg_speed_knots"],
                            "vessel_a_draught_change": left["draught_change_meters"],
                            "vessel_b_draught_change": right["draught_change_meters"],
                            "sts_score": min(max(sts_score, 0.0), 1.0),
                        }
                    )

        if not candidates:
            return pd.DataFrame(
                columns=[
                    "vessel_a_mmsi",
                    "vessel_b_mmsi",
                    "vessel_a_name",
                    "vessel_b_name",
                    "zone",
                    "overlap_start",
                    "overlap_end",
                    "overlap_seconds",
                    "overlap_hours",
                    "avg_distance_nm",
                    "min_distance_nm",
                    "vessel_a_avg_speed",
                    "vessel_b_avg_speed",
                    "vessel_a_draught_change",
                    "vessel_b_draught_change",
                    "sts_score",
                ]
            )

        return pd.DataFrame(candidates).sort_values("sts_score", ascending=False).reset_index(drop=True)

    def build_loitering_candidates(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame()

        working = events.copy()
        working["duration_hours"] = pd.to_numeric(working["duration_hours"], errors="coerce").fillna(0.0)
        working["avg_speed_knots"] = pd.to_numeric(working["avg_speed_knots"], errors="coerce")
        working["observations_count"] = pd.to_numeric(working["observations_count"], errors="coerce").fillna(0)
        working["draught_change_meters"] = pd.to_numeric(
            working["draught_change_meters"], errors="coerce"
        ).fillna(0.0)

        working["zone_duration_median_hours"] = (
            working.groupby("zone", dropna=False)["duration_hours"].transform("median").fillna(0.0)
        )
        working["zone_avg_speed_median_knots"] = (
            working.groupby("zone", dropna=False)["avg_speed_knots"].transform("median")
        )
        working["duration_vs_zone_median"] = working["duration_hours"] / working[
            "zone_duration_median_hours"
        ].clip(lower=0.1)

        speed_component = 1.0 - (
            working["avg_speed_knots"].fillna(0.0).clip(lower=0.0, upper=5.0) / 5.0
        )
        duration_component = ((working["duration_vs_zone_median"] - 1.0) / 2.0).clip(lower=0.0, upper=1.0)
        density_component = (working["observations_count"].clip(lower=1) / 20.0).clip(upper=1.0)

        working["loitering_score"] = (
            0.5 * duration_component + 0.35 * speed_component + 0.15 * density_component
        ).clip(lower=0.0, upper=1.0)
        working["is_low_speed"] = working["avg_speed_knots"].fillna(0.0) <= 2.0
        working["is_long_duration"] = working["duration_vs_zone_median"] >= 1.5

        columns = [
            "mmsi",
            "imo",
            "name",
            "zone",
            "entry_datetime",
            "exit_datetime",
            "status",
            "duration_seconds",
            "duration_hours",
            "avg_speed_knots",
            "observations_count",
            "draught_change_meters",
            "zone_duration_median_hours",
            "zone_avg_speed_median_knots",
            "duration_vs_zone_median",
            "is_low_speed",
            "is_long_duration",
            "loitering_score",
        ]
        return working.loc[:, columns].sort_values(
            ["loitering_score", "duration_hours"],
            ascending=[False, False],
        ).reset_index(drop=True)

    def build_zone_congestion(self, observations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if observations.empty:
            return pd.DataFrame()

        zone_obs = observations[observations["zone"].notna()].copy()
        if zone_obs.empty:
            return pd.DataFrame()

        zone_obs["observed_hour"] = zone_obs["observed_at"].dt.floor("h")
        congestion = (
            zone_obs.groupby(["zone", "observed_hour"], dropna=False)
            .agg(
                observation_count=("observed_at", "size"),
                unique_vessels=("mmsi", "nunique"),
                avg_speed_knots=("speed_knots", "mean"),
                median_speed_knots=("speed_knots", "median"),
            )
            .reset_index()
        )

        if not events.empty:
            events_working = events.copy()
            events_working["entry_hour"] = pd.to_datetime(
                events_working["entry_datetime"], utc=True, errors="coerce"
            ).dt.floor("h")
            event_hourly = (
                events_working.groupby(["zone", "entry_hour"], dropna=False)
                .agg(
                    event_count=("mmsi", "size"),
                    avg_event_duration_hours=("duration_hours", "mean"),
                    median_event_duration_hours=("duration_hours", "median"),
                    active_events=("status", lambda values: int((values == "active").sum())),
                )
                .reset_index()
                .rename(columns={"entry_hour": "observed_hour"})
            )
            congestion = congestion.merge(
                event_hourly,
                on=["zone", "observed_hour"],
                how="left",
            )

        fill_zero_columns = [
            "event_count",
            "avg_event_duration_hours",
            "median_event_duration_hours",
            "active_events",
        ]
        for column in fill_zero_columns:
            if column not in congestion.columns:
                congestion[column] = 0.0
        congestion[fill_zero_columns] = congestion[fill_zero_columns].fillna(0.0)

        vessel_component = (congestion["unique_vessels"] / max(congestion["unique_vessels"].max(), 1)).clip(0, 1)
        observation_component = (
            congestion["observation_count"] / max(congestion["observation_count"].max(), 1)
        ).clip(0, 1)
        slow_speed_component = 1.0 - (
            congestion["avg_speed_knots"].fillna(0.0).clip(lower=0.0, upper=10.0) / 10.0
        )
        congestion["congestion_score"] = (
            0.45 * vessel_component + 0.35 * observation_component + 0.20 * slow_speed_component
        ).clip(lower=0.0, upper=1.0)

        return congestion.sort_values(
            ["congestion_score", "unique_vessels", "observation_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def write_outputs(
        self,
        observations: pd.DataFrame,
        events: pd.DataFrame,
        sts_candidates: pd.DataFrame,
        loitering_candidates: pd.DataFrame,
        zone_congestion: pd.DataFrame,
        output_dir: str | Path,
        output_format: str = "csv",
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_format = output_format.lower()
        if output_format not in {"csv", "parquet"}:
            raise ValueError("output_format must be either 'csv' or 'parquet'")

        outputs: dict[str, Path] = {}
        parquet_available = output_format != "parquet" or self._parquet_engine_available()
        effective_format = output_format if parquet_available else "csv"

        for key, table_name in self.OUTPUT_TABLES.items():
            df = {
                "observations": observations,
                "events": events,
                "sts": sts_candidates,
                "loitering": loitering_candidates,
                "congestion": zone_congestion,
            }[key]
            prepared_df = self._prepare_for_output(df)
            suffix = "parquet" if effective_format == "parquet" else "csv"
            path = output_dir / f"{table_name}.{suffix}"
            outputs[key] = path

            if effective_format == "parquet":
                prepared_df.to_parquet(path, index=False)
            else:
                prepared_df.to_csv(path, index=False, encoding="utf-8")

        return outputs

    def build_summary(
        self,
        observations: pd.DataFrame,
        events: pd.DataFrame,
        sts_candidates: pd.DataFrame,
        loitering_candidates: pd.DataFrame,
        zone_congestion: pd.DataFrame,
    ) -> dict:
        missing_columns = [
            "observed_at",
            "mmsi",
            "imo",
            "latitude",
            "longitude",
            "speed_knots",
            "draught_meters",
            "zone",
        ]

        summary = {
            "unique_vessels": int(observations["mmsi"].nunique()) if not observations.empty else 0,
            "ais_observations": int(len(observations)),
            "zone_events": int(len(events)),
            "active_events": int((events["status"] == "active").sum()) if not events.empty else 0,
            "completed_events": int((events["status"] == "completed").sum()) if not events.empty else 0,
            "sts_candidates": int(len(sts_candidates)),
            "loitering_candidates": int(len(loitering_candidates)),
            "zone_congestion_windows": int(len(zone_congestion)),
            "missing_by_column": {
                column: int(observations[column].isna().sum()) if column in observations.columns else None
                for column in missing_columns
            },
        }
        return summary

    def _prepare_for_output(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        prepared = df.copy()
        for column in ["entry_datetime", "exit_datetime", "overlap_start", "overlap_end", "observed_at"]:
            if column in prepared.columns:
                prepared[column] = pd.to_datetime(prepared[column], utc=True, errors="coerce")
        datetime_columns = prepared.select_dtypes(
            include=["datetime64[ns, UTC]", "datetimetz"]
        ).columns
        for column in datetime_columns:
            prepared[column] = prepared[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return prepared

    @staticmethod
    def _parquet_engine_available() -> bool:
        try:
            import pyarrow  # noqa: F401

            return True
        except ImportError:
            pass

        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False

    def _discover_files(self, input_path: Path) -> list[Path]:
        if input_path.is_file():
            return [input_path]
        if input_path.is_dir():
            return sorted(
                [
                    file_path
                    for file_path in input_path.rglob("*")
                    if file_path.suffix.lower() in {".json", ".jsonl"}
                ]
            )
        raise FileNotFoundError(f"Input path was not found: {input_path}")

    def _load_records_from_url(self, url: str) -> list[dict]:
        request = Request(
            url,
            headers={
                "User-Agent": "lng-ml-research/0.1 (+https://tankermap.com/api/vessels/live)"
            },
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
            return [payload]
        return []

    def _load_records_from_file(self, file_path: Path) -> list[dict]:
        if file_path.suffix.lower() == ".jsonl":
            records = []
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
            return [payload]
        return []

    @staticmethod
    def _is_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    def _detect_zone(self, latitude: float, longitude: float) -> str | None:
        if pd.isna(latitude) or pd.isna(longitude):
            return None
        for zone in self.zones:
            if (
                zone.lat_min <= latitude <= zone.lat_max
                and zone.lon_min <= longitude <= zone.lon_max
            ):
                return zone.name
        return None

    def _pair_distance_stats(
        self,
        observations: pd.DataFrame,
        zone: str,
        vessel_a_mmsi: float,
        vessel_b_mmsi: float,
        overlap_start: pd.Timestamp,
        overlap_end: pd.Timestamp,
    ) -> dict[str, float | None]:
        vessel_a = observations[
            (observations["mmsi"] == vessel_a_mmsi)
            & (observations["zone"] == zone)
            & (observations["observed_at"] >= overlap_start)
            & (observations["observed_at"] <= overlap_end)
        ][["observed_at", "latitude", "longitude"]].dropna()
        vessel_b = observations[
            (observations["mmsi"] == vessel_b_mmsi)
            & (observations["zone"] == zone)
            & (observations["observed_at"] >= overlap_start)
            & (observations["observed_at"] <= overlap_end)
        ][["observed_at", "latitude", "longitude"]].dropna()

        if vessel_a.empty or vessel_b.empty:
            return {"avg_distance_nm": None, "min_distance_nm": None}

        distances: list[float] = []
        right_rows = list(vessel_b.itertuples(index=False))
        for left_row in vessel_a.itertuples(index=False):
            nearest = min(
                right_rows,
                key=lambda row: abs(row.observed_at - left_row.observed_at),
            )
            if abs(nearest.observed_at - left_row.observed_at) > self.distance_tolerance:
                continue
            distances.append(
                self._haversine_nm(
                    left_row.latitude,
                    left_row.longitude,
                    nearest.latitude,
                    nearest.longitude,
                )
            )

        if not distances:
            return {"avg_distance_nm": None, "min_distance_nm": None}

        return {
            "avg_distance_nm": float(sum(distances) / len(distances)),
            "min_distance_nm": float(min(distances)),
        }

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

    @staticmethod
    def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_km = 6371.0
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        hav = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        km = 2 * radius_km * asin(sqrt(hav))
        return km * 0.539957
