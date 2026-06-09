from datetime import date
from pathlib import Path

import polars as pl

from src.cohorts.t2dm_attrition import build_t2dm_attrition_table


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_build_t2dm_attrition_table_counts_steps(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    reports_dir = tmp_path / "reports"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {"patient_id": "p1"},
            {"patient_id": "p2"},
            {"patient_id": "p3"},
            {"patient_id": "p4"},
        ],
    )

    write_parquet(
        gold_dir / "t2dm_index_cohort.parquet",
        [
            {
                "patient_id": "p1",
                "index_date": date(2020, 1, 1),
                "age_at_index": 50,
                "meets_adult_age_rule": True,
                "observable_history_days_before_index": 730,
                "meets_baseline_history_rule": True,
                "has_prior_t1dm_before_index": False,
                "has_pregnancy_window_before_or_at_index": False,
                "meets_final_phase1_rules": True,
            },
            {
                "patient_id": "p2",
                "index_date": date(2020, 1, 1),
                "age_at_index": 50,
                "meets_adult_age_rule": True,
                "observable_history_days_before_index": 100,
                "meets_baseline_history_rule": False,
                "has_prior_t1dm_before_index": False,
                "has_pregnancy_window_before_or_at_index": False,
                "meets_final_phase1_rules": False,
            },
            {
                "patient_id": "p3",
                "index_date": date(2020, 1, 1),
                "age_at_index": 50,
                "meets_adult_age_rule": True,
                "observable_history_days_before_index": 730,
                "meets_baseline_history_rule": True,
                "has_prior_t1dm_before_index": True,
                "has_pregnancy_window_before_or_at_index": False,
                "meets_final_phase1_rules": False,
            },
        ],
    )

    parquet_output = gold_dir / "t2dm_attrition.parquet"
    csv_output = reports_dir / "t2dm_attrition.csv"

    manifest = build_t2dm_attrition_table(
        silver_dir=silver_dir,
        index_cohort_path=gold_dir / "t2dm_index_cohort.parquet",
        parquet_output_path=parquet_output,
        csv_output_path=csv_output,
    )

    df = pl.read_parquet(parquet_output)

    assert parquet_output.exists()
    assert csv_output.exists()

    assert df.height == 7
    assert df["remaining_n"].to_list() == [4, 3, 3, 2, 1, 1, 1]
    assert df["excluded_n"].to_list() == [0, 1, 0, 1, 1, 0, 0]

    assert manifest["summary"]["raw_patients_n"] == 4
    assert manifest["summary"]["t2dm_patients_n"] == 3
    assert manifest["summary"]["adult_t2dm_patients_n"] == 3
    assert manifest["summary"]["baseline_eligible_patients_n"] == 2
    assert manifest["summary"]["after_t1dm_exclusion_n"] == 1
    assert manifest["summary"]["after_pregnancy_exclusion_n"] == 1
    assert manifest["summary"]["final_cohort_n"] == 1


def test_build_t2dm_attrition_table_handles_no_t2dm_patients(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {"patient_id": "p1"},
            {"patient_id": "p2"},
        ],
    )

    empty_index = pl.DataFrame(
        schema={
            "patient_id": pl.Utf8,
            "index_date": pl.Date,
            "age_at_index": pl.Int64,
            "meets_adult_age_rule": pl.Boolean,
            "observable_history_days_before_index": pl.Int64,
            "meets_baseline_history_rule": pl.Boolean,
            "has_prior_t1dm_before_index": pl.Boolean,
            "has_pregnancy_window_before_or_at_index": pl.Boolean,
            "meets_final_phase1_rules": pl.Boolean,
        }
    )
    gold_dir.mkdir(parents=True, exist_ok=True)
    empty_index.write_parquet(gold_dir / "t2dm_index_cohort.parquet")

    parquet_output = gold_dir / "t2dm_attrition.parquet"

    manifest = build_t2dm_attrition_table(
        silver_dir=silver_dir,
        index_cohort_path=gold_dir / "t2dm_index_cohort.parquet",
        parquet_output_path=parquet_output,
    )

    df = pl.read_parquet(parquet_output)

    assert df["remaining_n"].to_list() == [2, 0, 0, 0, 0, 0, 0]
    assert df["excluded_n"].to_list() == [0, 2, 0, 0, 0, 0, 0]
    assert manifest["summary"]["t2dm_patients_n"] == 0
    assert manifest["summary"]["adult_t2dm_patients_n"] == 0
    assert manifest["summary"]["baseline_eligible_patients_n"] == 0
    assert manifest["summary"]["final_cohort_n"] == 0
