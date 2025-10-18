"""Command line interface for the homolog template benchmark."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import click

from .config import Split, load_config
from .pipeline import BenchmarkPipeline

_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def _configure_logging(level: str) -> None:
    resolved_level = _LOG_LEVELS.get(level.lower(), logging.INFO)
    logging.basicConfig(level=resolved_level, format="%(asctime)s - %(levelname)s - %(message)s")


def _parse_splits(values: Iterable[str]) -> Optional[Iterable[Split]]:
    selections = list(values)
    if not selections:
        return None
    return [Split(value) for value in selections]


@click.group()
@click.option("--log-level", default="info", help="Logging verbosity (debug, info, warning).")
def cli(log_level: str) -> None:
    """Entry point for the hypothesis test utilities."""

    _configure_logging(log_level)


@cli.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--split", "splits", type=click.Choice([choice.value for choice in Split]), multiple=True)
@click.option("--pilot", is_flag=True, help="Run only the pilot subset.")
@click.option("--include-templates", is_flag=True, help="Attempt the template-augmented Minifold run.")
@click.option(
    "--include-faplm",
    is_flag=True,
    help="Run the Minifold variant configured for FAPLM when enabled.",
)
@click.option(
    "--include-ism",
    is_flag=True,
    help="Run the Minifold variant configured for ISM when enabled.",
)
@click.option(
    "--include-baselines",
    is_flag=True,
    help="Run additional baseline predictors when enabled in the configuration.",
)
def run_command(
    config_path: Path,
    splits: Iterable[str],
    pilot: bool,
    include_templates: bool,
    include_faplm: bool,
    include_ism: bool,
    include_baselines: bool,
) -> None:
    """Execute Minifold inference for the requested dataset partitions."""

    config = load_config(config_path)
    pipeline = BenchmarkPipeline(config)

    if pilot:
        outputs = pipeline.pilot(
            include_templates=include_templates,
            include_faplm=include_faplm,
            include_ism=include_ism,
            include_baselines=include_baselines,
        )
    else:
        split_choices = _parse_splits(splits)
        outputs = pipeline.full(
            splits=split_choices,
            include_templates=include_templates,
            include_faplm=include_faplm,
            include_ism=include_ism,
            include_baselines=include_baselines,
        )

    click.echo("Outputs:")
    for name, path in outputs.items():
        click.echo(f"  {name}: {path}")


@cli.command("prepare-templates")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--split", "splits", type=click.Choice([choice.value for choice in Split]), multiple=True)
@click.option("--pilot", is_flag=True, help="Prepare metadata for only the pilot subset.")
def prepare_templates(config_path: Path, splits: Iterable[str], pilot: bool) -> None:
    """Create placeholder template metadata files for the requested targets."""

    config = load_config(config_path)
    pipeline = BenchmarkPipeline(config)

    if pilot:
        targets = pipeline._resolve_targets(pilot=True)  # noqa: SLF001 - deliberate use for CLI convenience.
    else:
        split_choices = _parse_splits(splits)
        targets = pipeline._resolve_targets(splits=split_choices)

    paths = pipeline.prepare_template_cache(targets)
    click.echo("Template metadata prepared:")
    for path in paths:
        click.echo(f"  {path}")


if __name__ == "__main__":
    cli()
