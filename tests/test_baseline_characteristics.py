from datetime import date
from pathlib import Path

import polars as pl

from src.analysis.baseline_characteristics import build_baseline_characteristics_table


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_build_baseline_characteristics_table(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    reports_dir = tmp_path / "reports"

    final_path = gold_dir / "t2dm_final_cohort.parquet"
    parquet_output = gold_dir / "baseline_characteristics.parquet"
    csv_output = reports_dir / "baseline_characteristics.csv"

    write_parquet(
        final_path,
        [
            {
                "patient_id": "p1",
                "index_date": date(2020, 1, 1),
                "age_at_index": 50,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
                "observable_history_days_before_index": 730,
            },
            {
                "patient_id": "p2",
                "index_date": date(2020, 1, 2),
                "age_at_index": 60,
                "sex": "F",
                "race": "black",
                "ethnicity": "nonhispanic",
                "observable_history_days_before_index": 365,
            },
        ],
    )

    manifest = build_baseline_characteristics_table(
        final_cohort_path=final_path,
        parquet_output_path=parquet_output,
        csv_output_path=csv_output,
    )

    table = pl.read_parquet(parquet_output)

    assert parquet_output.exists()
    assert csv_output.exists()

    assert manifest["summary"]["cohort_n"] == 2
    assert manifest["summary"]["baseline_table_rows"] == table.height
    assert manifest["summary"]["mean_age_at_index"] == 55

    cohort_n_row = table.filter(
        (pl.col("section") == "Cohort")
        & (pl.col("variable") == "Final analytical cohort")
        & (pl.col("statistic") == "N")
    )

    assert cohort_n_row.height == 1
    assert cohort_n_row["value"][0] == "2"

    age_mean_row = table.filter(
        (pl.col("variable") == "Age at index, years")
        & (pl.col("statistic") == "Mean (SD)")
    )

    assert age_mean_row.height == 1
    assert age_mean_row["value"][0].startswith("55.0")

    sex_rows = table.filter(pl.col("variable") == "Sex")
    assert sex_rows.height == 2


def test_build_baseline_characteristics_table_handles_empty_cohort(tmp_path: Path):
    gold_dir = tmp_path / "gold"

    final_path = gold_dir / "t2dm_final_cohort.parquet"
    parquet_output = gold_dir / "baseline_characteristics.parquet"

    empty = pl.DataFrame(
        schema={
            "patient_id": pl.Utf8,
            "index_date": pl.Date,
            "age_at_index": pl.Int64,
            "sex": pl.Utf8,
            "race": pl.Utf8,
            "ethnicity": pl.Utf8,
            "observable_history_days_before_index": pl.Int64,
        }
    )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    empty.write_parquet(final_path)

    manifest = build_baseline_characteristics_table(
        final_cohort_path=final_path,
        parquet_output_path=parquet_output,
    )

    table = pl.read_parquet(parquet_output)

    assert table.height > 0
    assert manifest["summary"]["cohort_n"] == 0
    assert manifest["summary"]["mean_age_at_index"] is None


def test_build_baseline_characteristics_requires_core_columns(tmp_path: Path):
    gold_dir = tmp_path / "gold"

    final_path = gold_dir / "t2dm_final_cohort.parquet"
    parquet_output = gold_dir / "baseline_characteristics.parquet"

    write_parquet(
        final_path,
        [
            {
                "patient_id": "p1",
                "age_at_index": 50,
            }
        ],
    )

    try:
        build_baseline_characteristics_table(
            final_cohort_path=final_path,
            parquet_output_path=parquet_output,
        )
    except ValueError as exc:
        assert "Final cohort missing columns required for baseline table" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing required columns")