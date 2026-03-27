import argparse
import json
from pathlib import Path

from lng_ml_research.ais_pipeline import AISDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AIS datasets for ML and STS analytics from JSON or JSONL inputs."
    )
    parser.add_argument(
        "--input",
        default="https://tankermap.com/api/vessels/live",
        help="Path to an AIS JSON/JSONL file, a directory with JSON/JSONL files, or a direct API URL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where output tables will be written.",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "parquet"],
        help="Output format for generated tables.",
    )
    parser.add_argument(
        "--event-gap-minutes",
        type=int,
        default=360,
        help="Gap after which observations are split into separate zone events.",
    )
    parser.add_argument(
        "--distance-tolerance-minutes",
        type=int,
        default=30,
        help="Maximum timestamp mismatch when estimating pair distance for STS candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = AISDatasetBuilder(
        event_gap_minutes=args.event_gap_minutes,
        distance_tolerance_minutes=args.distance_tolerance_minutes,
    )

    observations, events, sts_candidates, loitering_candidates, zone_congestion, summary = builder.build_all(
        args.input
    )
    outputs = builder.write_outputs(
        observations=observations,
        events=events,
        sts_candidates=sts_candidates,
        loitering_candidates=loitering_candidates,
        zone_congestion=zone_congestion,
        output_dir=Path(args.output_dir),
        output_format=args.format,
    )

    print("Output files:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
