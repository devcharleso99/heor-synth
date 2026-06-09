from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml


app = typer.Typer(
    help="HEOR Synth Phase 1 command-line interface."
)


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingest.raw_inventory import build_raw_inventory_from_scenario
from src.ingest.raw_to_bronze import build_bronze_from_scenario
from src.ingest.bronze_qa import build_bronze_qa_from_scenario
from src.normalize.bronze_to_silver import build_silver_from_scenario
from src.normalize.silver_qa import build_silver_qa_from_scenario
from src.cohorts.t2dm_index import build_t2dm_index_from_scenario
from src.cohorts.t2dm_attrition import build_t2dm_attrition_from_scenario
from src.cohorts.t2dm_final import build_t2dm_final_from_scenario
from src.analysis.baseline_characteristics import build_baseline_characteristics_from_scenario


REQUIRED_DIRECTORIES = [
    "configs/study",
    "configs/concepts",
    "configs/scenarios",
    "data/raw",
    "data/bronze",
    "data/silver",
    "data/gold",
    "sql",
    "src/ingest",
    "src/normalize",
    "src/omop_lite",
    "src/cohorts",
    "src/features",
    "src/analysis",
    "src/economics",
    "src/reporting",
    "src/cli",
    "tests",
    "reports",
    "docs",
    "notebooks",
]


REQUIRED_CONFIG_FILES = [
    "configs/study/t2dm_phase1.yaml",
    "configs/concepts/t2dm_concepts.yaml",
    "configs/scenarios/default_synthea.yaml",
]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file did not parse into a dictionary: {path}")

    return data


def check_required_directories() -> list[str]:
    missing = []

    for relative_path in REQUIRED_DIRECTORIES:
        path = REPO_ROOT / relative_path
        if not path.exists() or not path.is_dir():
            missing.append(relative_path)

    return missing


def check_required_config_files() -> list[str]:
    missing = []

    for relative_path in REQUIRED_CONFIG_FILES:
        path = REPO_ROOT / relative_path
        if not path.exists() or not path.is_file():
            missing.append(relative_path)

    return missing


def validate_study_config(data: dict[str, Any]) -> list[str]:
    errors = []

    required_top_level_keys = [
        "study",
        "objective",
        "population",
        "windows",
        "inclusion_criteria",
        "exclusion_criteria",
        "phase1_outputs",
        "analysis_language",
        "reproducibility",
    ]

    for key in required_top_level_keys:
        if key not in data:
            errors.append(f"study config missing top-level key: {key}")

    study = data.get("study", {})
    if study.get("id") != "t2dm_phase1":
        errors.append("study.id should be t2dm_phase1")

    population = data.get("population", {})
    if population.get("minimum_age_years") != 18:
        errors.append("population.minimum_age_years should be 18")

    windows = data.get("windows", {})
    baseline = windows.get("baseline", {})
    if baseline.get("days_before_index") != 365:
        errors.append("windows.baseline.days_before_index should be 365")

    inclusions = data.get("inclusion_criteria", [])
    exclusions = data.get("exclusion_criteria", [])

    if not isinstance(inclusions, list) or len(inclusions) < 4:
        errors.append("inclusion_criteria should contain at least 4 criteria")

    if not isinstance(exclusions, list) or len(exclusions) < 2:
        errors.append("exclusion_criteria should contain at least 2 criteria")

    return errors


def validate_concepts_config(data: dict[str, Any]) -> list[str]:
    errors = []

    concept_sets = data.get("concept_sets")
    if not isinstance(concept_sets, dict):
        return ["concepts config missing concept_sets dictionary"]

    required_concepts = [
        "type_2_diabetes_mellitus",
        "type_1_diabetes_mellitus",
        "pregnancy",
        "metformin",
        "sglt2_inhibitors",
        "glp1_receptor_agonists",
    ]

    for concept_name in required_concepts:
        if concept_name not in concept_sets:
            errors.append(f"concept_sets missing required concept: {concept_name}")

    t2dm = concept_sets.get("type_2_diabetes_mellitus", {})
    t2dm_codes = t2dm.get("codes", [])

    if not any(str(code.get("code")) == "44054006" for code in t2dm_codes):
        errors.append("type_2_diabetes_mellitus should include SNOMED code 44054006")

    return errors


def validate_scenario_config(data: dict[str, Any]) -> list[str]:
    errors = []

    required_top_level_keys = [
        "scenario",
        "synthea",
        "paths",
        "expected_raw_tables",
        "pipeline",
        "quality_checks",
    ]

    for key in required_top_level_keys:
        if key not in data:
            errors.append(f"scenario config missing top-level key: {key}")

    synthea = data.get("synthea", {})
    if "seed" not in synthea:
        errors.append("synthea.seed is required")

    if "reference_date" not in synthea:
        errors.append("synthea.reference_date is required")

    expected_raw_tables = data.get("expected_raw_tables", {})
    required_tables = expected_raw_tables.get("required", [])

    expected_required_tables = [
        "patients.csv",
        "encounters.csv",
        "conditions.csv",
        "medications.csv",
        "observations.csv",
        "procedures.csv",
        "careplans.csv",
    ]

    for table in expected_required_tables:
        if table not in required_tables:
            errors.append(f"expected_raw_tables.required missing: {table}")

    return errors


@app.command("tree-check")
def tree_check() -> None:
    """Validate that the expected project folders exist."""
    missing_dirs = check_required_directories()

    if missing_dirs:
        typer.echo("Missing required directories:")
        for item in missing_dirs:
            typer.echo(f" - {item}")
        raise typer.Exit(code=1)

    typer.echo("Repo tree check passed.")


@app.command("config-check")
def config_check() -> None:
    """Validate the Phase 1 YAML config files."""
    missing_files = check_required_config_files()

    if missing_files:
        typer.echo("Missing required config files:")
        for item in missing_files:
            typer.echo(f" - {item}")
        raise typer.Exit(code=1)

    study_config = load_yaml(REPO_ROOT / "configs/study/t2dm_phase1.yaml")
    concepts_config = load_yaml(REPO_ROOT / "configs/concepts/t2dm_concepts.yaml")
    scenario_config = load_yaml(REPO_ROOT / "configs/scenarios/default_synthea.yaml")

    errors = []
    errors.extend(validate_study_config(study_config))
    errors.extend(validate_concepts_config(concepts_config))
    errors.extend(validate_scenario_config(scenario_config))

    if errors:
        typer.echo("Config validation failed:")
        for error in errors:
            typer.echo(f" - {error}")
        raise typer.Exit(code=1)

    typer.echo("Config validation passed.")


@app.command("show-study")
def show_study() -> None:
    """Print the active study configuration summary."""
    study_config = load_yaml(REPO_ROOT / "configs/study/t2dm_phase1.yaml")
    scenario_config = load_yaml(REPO_ROOT / "configs/scenarios/default_synthea.yaml")

    summary = {
        "study_id": study_config["study"]["id"],
        "study_name": study_config["study"]["name"],
        "design": study_config["study"]["design"],
        "disease_area": study_config["study"]["disease_area"],
        "minimum_age_years": study_config["population"]["minimum_age_years"],
        "baseline_days_before_index": study_config["windows"]["baseline"]["days_before_index"],
        "follow_up_max_months": study_config["windows"]["follow_up"]["max_months_after_index"],
        "synthea_seed": scenario_config["synthea"]["seed"],
        "synthea_reference_date": scenario_config["synthea"]["reference_date"],
        "synthea_population_size": scenario_config["synthea"]["population_size"],
    }

    typer.echo(json.dumps(summary, indent=2))


@app.command("raw-inventory")
def raw_inventory(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output path for the raw inventory manifest JSON.",
    ),
    fail_on_missing: bool = typer.Option(
        False,
        "--fail-on-missing/--allow-missing",
        help="Fail with exit code 1 if required raw tables are missing.",
    ),
) -> None:
    """Inventory raw Synthea CSV files and write a reproducibility manifest."""
    manifest = build_raw_inventory_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        output_path=output_path,
    )

    summary = manifest["summary"]

    typer.echo("Raw inventory complete.")
    typer.echo(f"Raw directory: {manifest['raw_dir']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Required tables: {summary['required_table_count']}")
    typer.echo(f"Present required tables: {summary['present_required_count']}")
    typer.echo(f"Missing required tables: {summary['missing_required_count']}")
    typer.echo(f"Unexpected CSV files: {summary['unexpected_csv_count']}")

    if manifest["missing_required_tables"]:
        typer.echo("")
        typer.echo("Missing required raw tables:")
        for filename in manifest["missing_required_tables"]:
            typer.echo(f" - {filename}")

    if fail_on_missing and not summary["all_required_present"]:
        raise typer.Exit(code=1)


@app.command("raw-to-bronze")
def raw_to_bronze(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output path for the bronze conversion manifest JSON.",
    ),
    fail_on_error: bool = typer.Option(
        True,
        "--fail-on-error/--allow-errors",
        help="Fail with exit code 1 if any CSV fails conversion.",
    ),
) -> None:
    """Convert raw Synthea CSV files into bronze Parquet files."""
    manifest = build_bronze_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        output_path=output_path,
    )

    summary = manifest["summary"]

    typer.echo("Raw to bronze conversion complete.")
    typer.echo(f"Raw directory: {manifest['raw_dir']}")
    typer.echo(f"Bronze directory: {manifest['bronze_dir']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"CSV files seen: {summary['csv_files_seen']}")
    typer.echo(f"Converted files: {summary['converted_count']}")
    typer.echo(f"Failed files: {summary['failed_count']}")

    if manifest["failed"]:
        typer.echo("")
        typer.echo("Failed conversions:")
        for item in manifest["failed"]:
            typer.echo(f" - {item['table_name']}: {item['error']}")

    if fail_on_error and not summary["all_converted"]:
        raise typer.Exit(code=1)


@app.command("bronze-qa")
def bronze_qa(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output path for the bronze QA manifest JSON.",
    ),
    fail_on_error: bool = typer.Option(
        True,
        "--fail-on-error/--allow-errors",
        help="Fail with exit code 1 if structural bronze QA checks fail.",
    ),
) -> None:
    """Run structural QA checks against bronze Parquet files."""
    manifest = build_bronze_qa_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        output_path=output_path,
    )

    summary = manifest["summary"]
    t2dm_signal = manifest["t2dm_signal"]

    typer.echo("Bronze QA complete.")
    typer.echo(f"Bronze directory: {manifest['bronze_dir']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Missing required tables: {summary['missing_required_table_count']}")
    typer.echo(f"Empty required tables: {summary['empty_required_table_count']}")
    typer.echo(
        f"Required tables missing core columns: {summary['required_tables_missing_core_columns_count']}"
    )
    typer.echo(f"Failed patient-reference tables: {summary['failed_patient_reference_table_count']}")
    typer.echo(f"Structural checks passed: {summary['structural_checks_passed']}")
    typer.echo("")
    typer.echo("T2DM signal check:")
    typer.echo(f" - Code: {t2dm_signal['t2dm_code']}")
    typer.echo(f" - T2DM condition rows: {t2dm_signal['t2dm_condition_row_count']}")
    typer.echo(f" - T2DM patients: {t2dm_signal['t2dm_patient_count']}")
    typer.echo(f" - Status: {t2dm_signal['status']}")

    if manifest["missing_required_tables"]:
        typer.echo("")
        typer.echo("Missing required tables:")
        for table_name in manifest["missing_required_tables"]:
            typer.echo(f" - {table_name}")

    if manifest["empty_required_tables"]:
        typer.echo("")
        typer.echo("Empty required tables:")
        for table_name in manifest["empty_required_tables"]:
            typer.echo(f" - {table_name}")

    if manifest["required_tables_missing_core_columns"]:
        typer.echo("")
        typer.echo("Required tables missing core columns:")
        for item in manifest["required_tables_missing_core_columns"]:
            typer.echo(f" - {item['table_name']}: {item['missing_core_columns']}")

    if manifest["failed_patient_reference_tables"]:
        typer.echo("")
        typer.echo("Tables with unknown patient references:")
        for table_name in manifest["failed_patient_reference_tables"]:
            typer.echo(f" - {table_name}")

    if fail_on_error and not summary["structural_checks_passed"]:
        raise typer.Exit(code=1)


@app.command("bronze-to-silver")
def bronze_to_silver(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output path for the silver normalization manifest JSON.",
    ),
    fail_on_error: bool = typer.Option(
        True,
        "--fail-on-error/--allow-errors",
        help="Fail with exit code 1 if any core table fails silver normalization.",
    ),
) -> None:
    """Normalize bronze Parquet files into silver analysis-ready Parquet files."""
    manifest = build_silver_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        output_path=output_path,
    )

    summary = manifest["summary"]

    typer.echo("Bronze to silver normalization complete.")
    typer.echo(f"Bronze directory: {manifest['bronze_dir']}")
    typer.echo(f"Silver directory: {manifest['silver_dir']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Requested tables: {summary['requested_table_count']}")
    typer.echo(f"Normalized tables: {summary['normalized_count']}")
    typer.echo(f"Missing tables: {summary['missing_count']}")
    typer.echo(f"Failed tables: {summary['failed_count']}")
    typer.echo(f"All normalized: {summary['all_normalized']}")

    if manifest["missing"]:
        typer.echo("")
        typer.echo("Missing bronze tables:")
        for table_name in manifest["missing"]:
            typer.echo(f" - {table_name}")

    if manifest["failed"]:
        typer.echo("")
        typer.echo("Failed normalizations:")
        for item in manifest["failed"]:
            typer.echo(f" - {item['table_name']}: {item['error']}")

    if fail_on_error and not summary["all_normalized"]:
        raise typer.Exit(code=1)


@app.command("silver-qa")
def silver_qa(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output path for the silver QA manifest JSON.",
    ),
    fail_on_error: bool = typer.Option(
        True,
        "--fail-on-error/--allow-errors",
        help="Fail with exit code 1 if structural silver QA checks fail.",
    ),
) -> None:
    """Run structural QA checks against silver Parquet files."""
    manifest = build_silver_qa_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        output_path=output_path,
    )

    summary = manifest["summary"]
    t2dm_signal = manifest["t2dm_signal"]

    typer.echo("Silver QA complete.")
    typer.echo(f"Silver directory: {manifest['silver_dir']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Missing required tables: {summary['missing_required_table_count']}")
    typer.echo(f"Empty required tables: {summary['empty_required_table_count']}")
    typer.echo(f"Tables missing required columns: {summary['tables_missing_required_columns_count']}")
    typer.echo(f"Tables with invalid date dtypes: {summary['tables_with_invalid_date_dtypes_count']}")
    typer.echo(f"Failed patient-reference tables: {summary['failed_patient_reference_table_count']}")
    typer.echo(f"Structural checks passed: {summary['structural_checks_passed']}")
    typer.echo("")
    typer.echo("T2DM signal check:")
    typer.echo(f" - Code: {t2dm_signal['t2dm_code']}")
    typer.echo(f" - T2DM condition rows: {t2dm_signal['t2dm_condition_row_count']}")
    typer.echo(f" - T2DM patients: {t2dm_signal['t2dm_patient_count']}")
    typer.echo(f" - Earliest T2DM date: {t2dm_signal['earliest_t2dm_date']}")
    typer.echo(f" - Latest T2DM date: {t2dm_signal['latest_t2dm_date']}")
    typer.echo(f" - Status: {t2dm_signal['status']}")

    if manifest["missing_required_tables"]:
        typer.echo("")
        typer.echo("Missing required silver tables:")
        for table_name in manifest["missing_required_tables"]:
            typer.echo(f" - {table_name}")

    if manifest["tables_missing_required_columns"]:
        typer.echo("")
        typer.echo("Tables missing required columns:")
        for item in manifest["tables_missing_required_columns"]:
            typer.echo(f" - {item['table_name']}: {item['missing_required_columns']}")

    if manifest["tables_with_invalid_date_dtypes"]:
        typer.echo("")
        typer.echo("Tables with invalid date dtypes:")
        for item in manifest["tables_with_invalid_date_dtypes"]:
            typer.echo(f" - {item['table_name']}")

    if manifest["failed_patient_reference_tables"]:
        typer.echo("")
        typer.echo("Tables with unknown patient references:")
        for table_name in manifest["failed_patient_reference_tables"]:
            typer.echo(f" - {table_name}")

    if fail_on_error and not summary["structural_checks_passed"]:
        raise typer.Exit(code=1)


@app.command("t2dm-index")
def t2dm_index(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output path for the T2DM index cohort Parquet file.",
    ),
    manifest_path: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional output path for the T2DM index cohort manifest JSON.",
    ),
) -> None:
    """Build the first T2DM index cohort from silver patients and conditions."""
    manifest = build_t2dm_index_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    summary = manifest["summary"]

    typer.echo("T2DM index cohort build complete.")
    typer.echo(f"Silver directory: {manifest['silver_dir']}")
    typer.echo(f"Output path: {manifest['output_path']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Total patients: {summary['patients_total']}")
    typer.echo(f"Total conditions: {summary['conditions_total']}")
    typer.echo(f"T2DM condition rows: {summary['t2dm_condition_rows']}")
    typer.echo(f"T2DM patients: {summary['t2dm_patient_count']}")
    typer.echo(f"Index cohort rows: {summary['index_cohort_rows']}")
    typer.echo(f"Adult index cohort rows: {summary['adult_index_cohort_rows']}")
    typer.echo(f"Underage index cohort rows: {summary['underage_index_cohort_rows']}")
    typer.echo(f"Baseline-history eligible rows: {summary['baseline_history_eligible_rows']}")
    typer.echo(f"Baseline-history ineligible rows: {summary['baseline_history_ineligible_rows']}")
    typer.echo(f"Prior T1DM excluded rows: {summary['prior_t1dm_excluded_rows']}")
    typer.echo(f"Pregnancy-window excluded rows: {summary['pregnancy_window_excluded_rows']}")
    typer.echo(f"Final Phase 1 eligible rows: {summary['final_phase1_eligible_rows']}")


@app.command("t2dm-attrition")
def t2dm_attrition(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    parquet_output_path: Path | None = typer.Option(
        None,
        "--parquet-output",
        help="Optional output path for the attrition Parquet file.",
    ),
    csv_output_path: Path | None = typer.Option(
        None,
        "--csv-output",
        help="Optional output path for the attrition CSV file.",
    ),
    manifest_path: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional output path for the attrition manifest JSON.",
    ),
) -> None:
    """Build the first T2DM attrition table."""
    manifest = build_t2dm_attrition_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        parquet_output_path=parquet_output_path,
        csv_output_path=csv_output_path,
        manifest_path=manifest_path,
    )

    summary = manifest["summary"]

    typer.echo("T2DM attrition table build complete.")
    typer.echo(f"Silver directory: {manifest['silver_dir']}")
    typer.echo(f"Index cohort path: {manifest['index_cohort_path']}")
    typer.echo(f"Parquet output: {manifest['parquet_output_path']}")
    typer.echo(f"CSV output: {manifest['csv_output_path']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Raw patients: {summary['raw_patients_n']}")
    typer.echo(f"T2DM patients: {summary['t2dm_patients_n']}")
    typer.echo(f"Adult T2DM patients: {summary['adult_t2dm_patients_n']}")
    typer.echo(f"Excluded without T2DM: {summary['excluded_no_t2dm_n']}")
    typer.echo(f"Excluded underage at index: {summary['excluded_underage_at_index_n']}")
    typer.echo(f"Baseline eligible patients: {summary['baseline_eligible_patients_n']}")
    typer.echo(
        f"Excluded insufficient baseline history: {summary['excluded_insufficient_baseline_history_n']}"
    )
    typer.echo(f"Excluded prior T1DM: {summary['excluded_prior_t1dm_n']}")
    typer.echo(f"Excluded pregnancy window: {summary['excluded_pregnancy_window_n']}")
    typer.echo(f"Final cohort N: {summary['final_cohort_n']}")



@app.command("t2dm-final")
def t2dm_final(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    parquet_output_path: Path | None = typer.Option(
        None,
        "--parquet-output",
        help="Optional output path for the final cohort Parquet file.",
    ),
    csv_output_path: Path | None = typer.Option(
        None,
        "--csv-output",
        help="Optional output path for the final cohort CSV file.",
    ),
    manifest_path: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional output path for the final cohort manifest JSON.",
    ),
) -> None:
    """Materialize the final Phase 1 T2DM analytical cohort."""
    manifest = build_t2dm_final_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        parquet_output_path=parquet_output_path,
        csv_output_path=csv_output_path,
        manifest_path=manifest_path,
    )

    summary = manifest["summary"]

    typer.echo("T2DM final cohort build complete.")
    typer.echo(f"Index cohort path: {manifest['index_cohort_path']}")
    typer.echo(f"Parquet output: {manifest['parquet_output_path']}")
    typer.echo(f"CSV output: {manifest['csv_output_path']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Index cohort rows: {summary['index_cohort_rows']}")
    typer.echo(f"Final cohort rows: {summary['final_cohort_rows']}")
    typer.echo(f"Excluded from final: {summary['excluded_from_final_rows']}")
    typer.echo(f"Minimum age at index: {summary['min_age_at_index']}")
    typer.echo(f"Maximum age at index: {summary['max_age_at_index']}")
    typer.echo(f"Mean age at index: {summary['mean_age_at_index']}")



@app.command("t2dm-baseline")
def t2dm_baseline(
    scenario_path: Path = typer.Option(
        REPO_ROOT / "configs/scenarios/default_synthea.yaml",
        "--scenario",
        help="Path to the Synthea scenario YAML file.",
    ),
    parquet_output_path: Path | None = typer.Option(
        None,
        "--parquet-output",
        help="Optional output path for the baseline characteristics Parquet file.",
    ),
    csv_output_path: Path | None = typer.Option(
        None,
        "--csv-output",
        help="Optional output path for the baseline characteristics CSV file.",
    ),
    manifest_path: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional output path for the baseline characteristics manifest JSON.",
    ),
) -> None:
    """Build the baseline characteristics table for the final T2DM cohort."""
    manifest = build_baseline_characteristics_from_scenario(
        repo_root=REPO_ROOT,
        scenario_path=scenario_path,
        parquet_output_path=parquet_output_path,
        csv_output_path=csv_output_path,
        manifest_path=manifest_path,
    )

    summary = manifest["summary"]

    typer.echo("T2DM baseline characteristics table build complete.")
    typer.echo(f"Final cohort path: {manifest['final_cohort_path']}")
    typer.echo(f"Parquet output: {manifest['parquet_output_path']}")
    typer.echo(f"CSV output: {manifest['csv_output_path']}")
    typer.echo(f"Manifest written: {manifest['manifest_path']}")
    typer.echo(f"Cohort N: {summary['cohort_n']}")
    typer.echo(f"Baseline table rows: {summary['baseline_table_rows']}")
    typer.echo(f"Mean age at index: {summary['mean_age_at_index']}")
    typer.echo(f"Minimum age at index: {summary['min_age_at_index']}")
    typer.echo(f"Maximum age at index: {summary['max_age_at_index']}")


@app.command("doctor")
def doctor() -> None:
    """Run all current project health checks."""
    missing_dirs = check_required_directories()
    missing_files = check_required_config_files()

    if missing_dirs or missing_files:
        if missing_dirs:
            typer.echo("Missing required directories:")
            for item in missing_dirs:
                typer.echo(f" - {item}")

        if missing_files:
            typer.echo("Missing required config files:")
            for item in missing_files:
                typer.echo(f" - {item}")

        raise typer.Exit(code=1)

    study_config = load_yaml(REPO_ROOT / "configs/study/t2dm_phase1.yaml")
    concepts_config = load_yaml(REPO_ROOT / "configs/concepts/t2dm_concepts.yaml")
    scenario_config = load_yaml(REPO_ROOT / "configs/scenarios/default_synthea.yaml")

    errors = []
    errors.extend(validate_study_config(study_config))
    errors.extend(validate_concepts_config(concepts_config))
    errors.extend(validate_scenario_config(scenario_config))

    if errors:
        typer.echo("Project doctor failed:")
        for error in errors:
            typer.echo(f" - {error}")
        raise typer.Exit(code=1)

    typer.echo("Project doctor passed.")
    typer.echo("Repo tree is valid.")
    typer.echo("Phase 1 YAML configs are valid.")
    typer.echo("Ready for raw-data planning.")


if __name__ == "__main__":
    app()
