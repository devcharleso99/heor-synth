from pathlib import Path

path = Path("src/cli/main.py")
text = path.read_text(encoding="utf-8")

old_import = "from src.cohorts.t2dm_final import build_t2dm_final_from_scenario\n"

new_import = (
    "from src.cohorts.t2dm_final import build_t2dm_final_from_scenario\n"
    "from src.analysis.baseline_characteristics import build_baseline_characteristics_from_scenario\n"
)

if "from src.analysis.baseline_characteristics import build_baseline_characteristics_from_scenario" not in text:
    text = text.replace(old_import, new_import)

baseline_command = '''
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
'''

if '@app.command("t2dm-baseline")' not in text:
    text = text.replace('@app.command("doctor")', baseline_command + '\n\n@app.command("doctor")')

path.write_text(text, encoding="utf-8")