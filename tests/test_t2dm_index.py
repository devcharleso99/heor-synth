from datetime import date
from pathlib import Path

import polars as pl

from src.cohorts.t2dm_index import build_t2dm_index_cohort


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_build_t2dm_index_cohort_selects_first_t2dm_date_and_baseline_history(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    output_path = tmp_path / "gold" / "t2dm_index_cohort.parquet"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(1970, 1, 1),
                "death_date": None,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
            }
        ],
    )

    write_parquet(
        silver_dir / "encounters.parquet",
        [
            {"patient_id": "p1", "encounter_start_date": date(2018, 1, 1)},
            {"patient_id": "p1", "encounter_start_date": date(2019, 1, 1)},
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e2",
                "condition_start_date": date(2021, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            },
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            },
        ],
    )

    manifest = build_t2dm_index_cohort(
        silver_dir=silver_dir,
        output_path=output_path,
    )

    df = pl.read_parquet(output_path)

    assert manifest["summary"]["t2dm_condition_rows"] == 2
    assert manifest["summary"]["t2dm_patient_count"] == 1
    assert manifest["summary"]["final_phase1_eligible_rows"] == 1
    assert df.height == 1
    assert df["patient_id"][0] == "p1"
    assert df["index_date"][0] == date(2020, 1, 1)
    assert df["index_encounter_id"][0] == "e1"
    assert df["age_at_index"][0] == 50
    assert df["observable_history_days_before_index"][0] == 730
    assert df["meets_adult_age_rule"][0] is True
    assert df["meets_baseline_history_rule"][0] is True
    assert df["has_prior_t1dm_before_index"][0] is False
    assert df["has_pregnancy_window_before_or_at_index"][0] is False
    assert df["meets_final_phase1_rules"][0] is True


def test_build_t2dm_index_cohort_flags_insufficient_baseline_history(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    output_path = tmp_path / "gold" / "t2dm_index_cohort.parquet"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(1970, 1, 1),
                "death_date": None,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
            }
        ],
    )

    write_parquet(
        silver_dir / "encounters.parquet",
        [
            {"patient_id": "p1", "encounter_start_date": date(2019, 9, 1)},
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            }
        ],
    )

    manifest = build_t2dm_index_cohort(
        silver_dir=silver_dir,
        output_path=output_path,
    )

    df = pl.read_parquet(output_path)

    assert df["observable_history_days_before_index"][0] < 365
    assert df["meets_baseline_history_rule"][0] is False
    assert df["meets_final_phase1_rules"][0] is False
    assert manifest["summary"]["baseline_history_eligible_rows"] == 0
    assert manifest["summary"]["baseline_history_ineligible_rows"] == 1


def test_build_t2dm_index_cohort_flags_underage_patients(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    output_path = tmp_path / "gold" / "t2dm_index_cohort.parquet"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(2010, 1, 1),
                "death_date": None,
                "sex": "F",
                "race": "white",
                "ethnicity": "nonhispanic",
            }
        ],
    )

    write_parquet(
        silver_dir / "encounters.parquet",
        [
            {"patient_id": "p1", "encounter_start_date": date(2018, 1, 1)},
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            }
        ],
    )

    manifest = build_t2dm_index_cohort(
        silver_dir=silver_dir,
        output_path=output_path,
    )

    df = pl.read_parquet(output_path)

    assert df["age_at_index"][0] == 10
    assert df["meets_adult_age_rule"][0] is False
    assert df["meets_final_phase1_rules"][0] is False
    assert manifest["summary"]["adult_index_cohort_rows"] == 0
    assert manifest["summary"]["underage_index_cohort_rows"] == 1


def test_build_t2dm_index_cohort_flags_prior_t1dm_exclusion(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    output_path = tmp_path / "gold" / "t2dm_index_cohort.parquet"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(1970, 1, 1),
                "death_date": None,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
            }
        ],
    )

    write_parquet(
        silver_dir / "encounters.parquet",
        [
            {"patient_id": "p1", "encounter_start_date": date(2018, 1, 1)},
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e0",
                "condition_start_date": date(2019, 1, 1),
                "condition_stop_date": None,
                "code": "46635009",
                "description": "Diabetes mellitus type 1",
                "source_table": "conditions",
            },
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            },
        ],
    )

    manifest = build_t2dm_index_cohort(
        silver_dir=silver_dir,
        output_path=output_path,
    )

    df = pl.read_parquet(output_path)

    assert df["has_prior_t1dm_before_index"][0] is True
    assert df["meets_prior_t1dm_exclusion_rule"][0] is False
    assert df["meets_final_phase1_rules"][0] is False
    assert manifest["summary"]["prior_t1dm_excluded_rows"] == 1
    assert manifest["summary"]["final_phase1_eligible_rows"] == 0


def test_build_t2dm_index_cohort_flags_pregnancy_exclusion(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    output_path = tmp_path / "gold" / "t2dm_index_cohort.parquet"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(1970, 1, 1),
                "death_date": None,
                "sex": "F",
                "race": "white",
                "ethnicity": "nonhispanic",
            }
        ],
    )

    write_parquet(
        silver_dir / "encounters.parquet",
        [
            {"patient_id": "p1", "encounter_start_date": date(2018, 1, 1)},
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e0",
                "condition_start_date": date(2019, 1, 1),
                "condition_stop_date": None,
                "code": "77386006",
                "description": "Pregnancy",
                "source_table": "conditions",
            },
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            },
        ],
    )

    manifest = build_t2dm_index_cohort(
        silver_dir=silver_dir,
        output_path=output_path,
    )

    df = pl.read_parquet(output_path)

    assert df["has_pregnancy_window_before_or_at_index"][0] is True
    assert df["meets_pregnancy_exclusion_rule"][0] is False
    assert df["meets_final_phase1_rules"][0] is False
    assert manifest["summary"]["pregnancy_window_excluded_rows"] == 1
    assert manifest["summary"]["final_phase1_eligible_rows"] == 0


def test_build_t2dm_index_cohort_ignores_non_t2dm_conditions(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    output_path = tmp_path / "gold" / "t2dm_index_cohort.parquet"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(1970, 1, 1),
                "death_date": None,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
            }
        ],
    )

    write_parquet(
        silver_dir / "encounters.parquet",
        [
            {"patient_id": "p1", "encounter_start_date": date(2018, 1, 1)},
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "123",
                "description": "Other condition",
                "source_table": "conditions",
            }
        ],
    )

    manifest = build_t2dm_index_cohort(
        silver_dir=silver_dir,
        output_path=output_path,
    )

    df = pl.read_parquet(output_path)

    assert df.height == 0
    assert manifest["summary"]["t2dm_condition_rows"] == 0
    assert manifest["summary"]["t2dm_patient_count"] == 0
    assert manifest["summary"]["index_cohort_rows"] == 0
