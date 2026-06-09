from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file did not parse into a dictionary: {path}")

    return data


def write_json(data: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path


def format_float(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return ""

    return f"{float(value):.{digits}f}"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return sorted_values[int(position)]

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = position - lower

    return lower_value + ((upper_value - lower_value) * weight)


def clean_level(value: Any) -> str:
    if value is None:
        return "Missing"

    text = str(value).strip()

    if text == "":
        return "Missing"

    return text


def make_row(
    section: str,
    variable: str,
    level: str,
    statistic: str,
    value: str,
    n: int | None,
    denominator: int | None,
    percent: float | None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "section": section,
        "variable": variable,
        "level": level,
        "statistic": statistic,
        "value": value,
        "n": n,
        "denominator": denominator,
        "percent": percent,
        "notes": notes,
    }


def numeric_summary_rows(
    df: pl.DataFrame,
    column: str,
    section: str,
    variable: str,
    total_n: int,
) -> list[dict[str, Any]]:
    if column not in df.columns:
        return [
            make_row(
                section=section,
                variable=variable,
                level="Overall",
                statistic="Unavailable",
                value="",
                n=None,
                denominator=total_n,
                percent=None,
                notes=f"Column not found: {column}",
            )
        ]

    raw_values = df[column].drop_nulls().to_list()
    values = [float(value) for value in raw_values if value is not None]

    if not values:
        return [
            make_row(
                section=section,
                variable=variable,
                level="Overall",
                statistic="No non-missing values",
                value="",
                n=0,
                denominator=total_n,
                percent=None,
            )
        ]

    mean_value = statistics.mean(values)
    sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
    median_value = statistics.median(values)
    q1_value = percentile(values, 0.25)
    q3_value = percentile(values, 0.75)
    min_value = min(values)
    max_value = max(values)

    return [
        make_row(
            section=section,
            variable=variable,
            level="Overall",
            statistic="Mean (SD)",
            value=f"{format_float(mean_value)} ({format_float(sd_value)})",
            n=len(values),
            denominator=total_n,
            percent=None,
        ),
        make_row(
            section=section,
            variable=variable,
            level="Overall",
            statistic="Median [Q1, Q3]",
            value=f"{format_float(median_value)} [{format_float(q1_value)}, {format_float(q3_value)}]",
            n=len(values),
            denominator=total_n,
            percent=None,
        ),
        make_row(
            section=section,
            variable=variable,
            level="Overall",
            statistic="Min, Max",
            value=f"{format_float(min_value)}, {format_float(max_value)}",
            n=len(values),
            denominator=total_n,
            percent=None,
        ),
    ]


def categorical_summary_rows(
    df: pl.DataFrame,
    column: str,
    section: str,
    variable: str,
    total_n: int,
) -> list[dict[str, Any]]:
    if column not in df.columns:
        return [
            make_row(
                section=section,
                variable=variable,
                level="Unavailable",
                statistic="Unavailable",
                value="",
                n=None,
                denominator=total_n,
                percent=None,
                notes=f"Column not found: {column}",
            )
        ]

    counts: dict[str, int] = {}

    for value in df[column].to_list():
        level = clean_level(value)
        counts[level] = counts.get(level, 0) + 1

    rows = []

    for level, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        percent = round((count / total_n) * 100, 1) if total_n > 0 else None
        percent_text = format_float(percent) if percent is not None else ""

        rows.append(
            make_row(
                section=section,
                variable=variable,
                level=level,
                statistic="n (%)",
                value=f"{count} ({percent_text}%)" if percent_text else str(count),
                n=count,
                denominator=total_n,
                percent=percent,
            )
        )

    if not rows:
        rows.append(
            make_row(
                section=section,
                variable=variable,
                level="No rows",
                statistic="n (%)",
                value="0",
                n=0,
                denominator=total_n,
                percent=None,
            )
        )

    return rows


def build_baseline_characteristics_table(
    final_cohort_path: Path,
    parquet_output_path: Path,
    csv_output_path: Path | None = None,
) -> dict[str, Any]:
    final_cohort_path = Path(final_cohort_path)
    parquet_output_path = Path(parquet_output_path)

    if not final_cohort_path.exists():
        raise FileNotFoundError(
            f"Missing final cohort file: {final_cohort_path}. "
            "Run the t2dm-final command first."
        )

    cohort = pl.read_parquet(final_cohort_path)

    required_columns = {
        "patient_id",
        "age_at_index",
        "sex",
        "race",
        "ethnicity",
        "observable_history_days_before_index",
        "index_date",
    }

    missing_columns = required_columns - set(cohort.columns)
    if missing_columns:
        raise ValueError(
            f"Final cohort missing columns required for baseline table: {sorted(missing_columns)}"
        )

    total_n = cohort.select("patient_id").unique().height if cohort.height > 0 else 0

    rows: list[dict[str, Any]] = [
        make_row(
            section="Cohort",
            variable="Final analytical cohort",
            level="Overall",
            statistic="N",
            value=str(total_n),
            n=total_n,
            denominator=total_n,
            percent=100.0 if total_n > 0 else None,
            notes="Patients meeting all Phase 1 cohort criteria.",
        )
    ]

    rows.extend(
        numeric_summary_rows(
            df=cohort,
            column="age_at_index",
            section="Demographics",
            variable="Age at index, years",
            total_n=total_n,
        )
    )

    rows.extend(
        categorical_summary_rows(
            df=cohort,
            column="sex",
            section="Demographics",
            variable="Sex",
            total_n=total_n,
        )
    )

    rows.extend(
        categorical_summary_rows(
            df=cohort,
            column="race",
            section="Demographics",
            variable="Race",
            total_n=total_n,
        )
    )

    rows.extend(
        categorical_summary_rows(
            df=cohort,
            column="ethnicity",
            section="Demographics",
            variable="Ethnicity",
            total_n=total_n,
        )
    )

    rows.extend(
        numeric_summary_rows(
            df=cohort,
            column="observable_history_days_before_index",
            section="Baseline observation",
            variable="Observable history before index, days",
            total_n=total_n,
        )
    )

    table = pl.DataFrame(rows)

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(parquet_output_path)

    csv_written = None
    if csv_output_path is not None:
        csv_output_path = Path(csv_output_path)
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(csv_output_path)
        csv_written = str(csv_output_path)

    age_values = cohort["age_at_index"].drop_nulls().to_list() if "age_at_index" in cohort.columns else []
    age_values = [float(value) for value in age_values if value is not None]

    manifest = {
        "manifest_type": "baseline_characteristics",
        "generated_at_utc": utc_now_iso(),
        "final_cohort_path": str(final_cohort_path),
        "parquet_output_path": str(parquet_output_path),
        "csv_output_path": csv_written,
        "summary": {
            "cohort_n": total_n,
            "baseline_table_rows": table.height,
            "mean_age_at_index": statistics.mean(age_values) if age_values else None,
            "min_age_at_index": min(age_values) if age_values else None,
            "max_age_at_index": max(age_values) if age_values else None,
        },
        "output_columns": table.columns,
    }

    return manifest


def build_baseline_characteristics_from_scenario(
    repo_root: Path,
    scenario_path: Path | str = "configs/scenarios/default_synthea.yaml",
    parquet_output_path: Path | str | None = None,
    csv_output_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)

    scenario_path = Path(scenario_path)
    if not scenario_path.is_absolute():
        scenario_path = repo_root / scenario_path

    scenario_config = load_yaml(scenario_path)

    gold_dir = Path(scenario_config["paths"]["gold_dir"])
    if not gold_dir.is_absolute():
        gold_dir = repo_root / gold_dir

    final_cohort_path = gold_dir / "t2dm_final_cohort.parquet"

    if parquet_output_path is None:
        parquet_output_path = gold_dir / "baseline_characteristics.parquet"
    else:
        parquet_output_path = Path(parquet_output_path)
        if not parquet_output_path.is_absolute():
            parquet_output_path = repo_root / parquet_output_path

    if csv_output_path is None:
        csv_output_path = repo_root / "reports" / "tables" / f"{scenario_config['scenario']['id']}_baseline_characteristics.csv"
    else:
        csv_output_path = Path(csv_output_path)
        if not csv_output_path.is_absolute():
            csv_output_path = repo_root / csv_output_path

    manifest = build_baseline_characteristics_table(
        final_cohort_path=final_cohort_path,
        parquet_output_path=parquet_output_path,
        csv_output_path=csv_output_path,
    )

    manifest["scenario"] = {
        "id": scenario_config["scenario"]["id"],
        "name": scenario_config["scenario"]["name"],
        "scenario_path": str(scenario_path),
    }

    manifest["synthea"] = scenario_config["synthea"]

    if manifest_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest_path = repo_root / "reports" / "manifests" / f"baseline_characteristics_{timestamp}.json"
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path

    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest, manifest_path)

    return manifest