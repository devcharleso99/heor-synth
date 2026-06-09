from datetime import date
from pathlib import Path

import sys


# Ensure repo root is importable so `import src...` works when running tests
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.cohorts.t2dm_index import build_t2dm_index_cohort


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_build_t2dm_index_cohort_selects_first_t2dm_date(tmp_path: Path):
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
    assert df.height == 1
    assert df["patient_id"][0] == "p1"
    assert df["index_date"][0] == date(2020, 1, 1)
    assert df["index_encounter_id"][0] == "e1"
    assert df["age_at_index"][0] == 50
    assert df["meets_adult_age_rule"][0] is True


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
    assert manifest["summary"]["adult_index_cohort_rows"] == 0
    assert manifest["summary"]["underage_index_cohort_rows"] == 1


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