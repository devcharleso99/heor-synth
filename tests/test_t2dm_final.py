from datetime import date
from pathlib import Path

import polars as pl

from src.cohorts.t2dm_final import build_t2dm_final_cohort


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_build_t2dm_final_cohort_filters_final_eligible_rows(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    reports_dir = tmp_path / "reports"

    index_path = gold_dir / "t2dm_index_cohort.parquet"
    parquet_output = gold_dir / "t2dm_final_cohort.parquet"
    csv_output = reports_dir / "t2dm_final_cohort.csv"

    write_parquet(
        index_path,
        [
            {
                "patient_id": "p1",
                "index_date": date(2020, 1, 1),
                "age_at_index": 50,
                "meets_adult_age_rule": True,
                "meets_baseline_history_rule": True,
                "meets_prior_t1dm_exclusion_rule": True,
                "meets_pregnancy_exclusion_rule": True,
                "meets_final_phase1_rules": True,
            },
            {
                "patient_id": "p2",
                "index_date": date(2020, 1, 2),
                "age_at_index": 55,
                "meets_adult_age_rule": True,
                "meets_baseline_history_rule": False,
                "meets_prior_t1dm_exclusion_rule": True,
                "meets_pregnancy_exclusion_rule": True,
                "meets_final_phase1_rules": False,
            },
        ],
    )

    manifest = build_t2dm_final_cohort(
        index_cohort_path=index_path,
        parquet_output_path=parquet_output,
        csv_output_path=csv_output,
    )

    df = pl.read_parquet(parquet_output)

    assert parquet_output.exists()
    assert csv_output.exists()
    assert df.height == 1
    assert df["patient_id"][0] == "p1"

    assert manifest["summary"]["index_cohort_rows"] == 2
    assert manifest["summary"]["final_cohort_rows"] == 1
    assert manifest["summary"]["excluded_from_final_rows"] == 1
    assert manifest["summary"]["min_age_at_index"] == 50
    assert manifest["summary"]["max_age_at_index"] == 50


def test_build_t2dm_final_cohort_handles_empty_final_cohort(tmp_path: Path):
    gold_dir = tmp_path / "gold"

    index_path = gold_dir / "t2dm_index_cohort.parquet"
    parquet_output = gold_dir / "t2dm_final_cohort.parquet"

    write_parquet(
        index_path,
        [
            {
                "patient_id": "p1",
                "index_date": date(2020, 1, 1),
                "age_at_index": 50,
                "meets_adult_age_rule": True,
                "meets_baseline_history_rule": False,
                "meets_prior_t1dm_exclusion_rule": True,
                "meets_pregnancy_exclusion_rule": True,
                "meets_final_phase1_rules": False,
            }
        ],
    )

    manifest = build_t2dm_final_cohort(
        index_cohort_path=index_path,
        parquet_output_path=parquet_output,
    )

    df = pl.read_parquet(parquet_output)

    assert df.height == 0
    assert manifest["summary"]["index_cohort_rows"] == 1
    assert manifest["summary"]["final_cohort_rows"] == 0
    assert manifest["summary"]["excluded_from_final_rows"] == 1
    assert manifest["summary"]["min_age_at_index"] is None


def test_build_t2dm_final_cohort_requires_final_rule_column(tmp_path: Path):
    gold_dir = tmp_path / "gold"

    index_path = gold_dir / "t2dm_index_cohort.parquet"
    parquet_output = gold_dir / "t2dm_final_cohort.parquet"

    write_parquet(
        index_path,
        [
            {
                "patient_id": "p1",
                "index_date": date(2020, 1, 1),
                "age_at_index": 50,
            }
        ],
    )

    try:
        build_t2dm_final_cohort(
            index_cohort_path=index_path,
            parquet_output_path=parquet_output,
        )
    except ValueError as exc:
        assert "missing columns required for final cohort" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing required columns")
