"""Quick sanity probe comparing Minifold with/without template distograms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import click
import torch
from Bio import SeqIO
from minifold.utils.template_probe import (
    TemplateResidueMap,
    align_template_to_target,
    bucket_templates_by_identity,
    build_distogram_from_templates,
    extract_ca_coordinates,
    load_template,
    protein_to_sequence,
)


def _load_target_sequence(path: Path) -> SeqIO.SeqRecord:
    records = list(SeqIO.parse(str(path), "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {path}.")
    if len(records) > 1:
        raise ValueError("Template quick test expects a single FASTA record.")
    return records[0]


def _resolve_template_paths(paths: Sequence[Path], directories: Sequence[Path]) -> List[Path]:
    resolved: List[Path] = []
    for explicit in paths:
        resolved.append(explicit)
    for directory in directories:
        for candidate in directory.iterdir():
            if candidate.suffix.lower() in {".pdb", ".cif"}:
                resolved.append(candidate)
    return resolved


def _prepare_template_candidates(
    target_sequence: str,
    template_paths: Iterable[Path],
    chain: str | None,
) -> List[TemplateResidueMap]:
    candidates: List[TemplateResidueMap] = []
    for template_path in template_paths:
        structure = load_template(template_path, chain_id=chain)
        template_sequence = protein_to_sequence(structure)
        identity, mapping = align_template_to_target(target_sequence, template_sequence)
        coordinates, mask = extract_ca_coordinates(structure)
        candidates.append(
            TemplateResidueMap(
                name=template_path.stem,
                coordinates=coordinates,
                mask=mask,
                mapping=mapping,
                identity=identity,
            )
        )
    return candidates


def _write_template_payload(
    output_dir: Path,
    sequence_id: str,
    dist: torch.Tensor,
) -> Path:
    payload = {sequence_id: {"distogram": dist.cpu()}}
    destination = output_dir / f"{sequence_id}_template_overrides.pt"
    torch.save(payload, destination)
    return destination


def _invoke_predict(
    fasta_path: Path,
    out_dir: Path,
    cache: Path,
    checkpoint: Path | None,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    compile_model: bool,
    use_kernels: bool,
    template_override: Path | None,
) -> None:
    from predict import predict as predict_command

    args = [
        str(fasta_path),
        "--out_dir",
        str(out_dir),
        "--cache",
        str(cache),
        "--model_size",
        model_size,
        "--token_per_batch",
        str(token_per_batch),
        "--num_recycling",
        str(num_recycling),
    ]

    if checkpoint is not None:
        args.extend(["--checkpoint", str(checkpoint)])
    if compile_model:
        args.append("--compile")
    if use_kernels:
        args.append("--kernels")
    if template_override is not None:
        args.extend(["--template_overrides", str(template_override)])

    predict_command.main(args=args, standalone_mode=False)


@click.command()
@click.argument("target_fasta", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--template",
    "templates",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="Explicit template structure files (PDB or CIF).",
)
@click.option(
    "--template-dir",
    "template_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    help="Directory containing template structures.",
)
@click.option(
    "--chain-id",
    default=None,
    help="Optional chain identifier to select from multi-chain templates.",
)
@click.option(
    "--identity-levels",
    default="0.9,0.8,0.7,0.6",
    show_default=True,
    help="Comma-separated sequence identity thresholds to evaluate.",
)
@click.option(
    "--max-templates-per-level",
    default=1,
    type=int,
    show_default=True,
    help="Maximum number of templates to use per identity bracket.",
)
@click.option(
    "--output-dir",
    default=Path("./proteinttt_quicktest"),
    type=click.Path(path_type=Path),
    help="Directory used to store intermediate files and predictions.",
)
@click.option(
    "--cache",
    default=Path("./minifold_cache"),
    type=click.Path(path_type=Path),
    help="Model cache directory passed through to the Minifold CLI.",
)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--model-size", default="48L", show_default=True)
@click.option("--token-per-batch", default=2048, show_default=True)
@click.option("--num-recycling", default=3, show_default=True)
@click.option("--compile/--no-compile", default=False, show_default=True)
@click.option("--kernels/--no-kernels", default=False, show_default=True)
def main(
    target_fasta: Path,
    templates: Sequence[Path],
    template_dirs: Sequence[Path],
    chain_id: str | None,
    identity_levels: str,
    max_templates_per_level: int,
    output_dir: Path,
    cache: Path,
    checkpoint: Path | None,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    compile: bool,
    kernels: bool,
) -> None:
    """Run Minifold with/without template distograms across identity brackets."""

    record = _load_target_sequence(target_fasta)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(level.strip()) for level in identity_levels.split(",") if level.strip()]
    template_paths = _resolve_template_paths(templates, template_dirs)
    if not template_paths:
        raise ValueError("No template structures were provided.")

    candidates = _prepare_template_candidates(str(record.seq), template_paths, chain_id)
    buckets = bucket_templates_by_identity(candidates, thresholds, limit_per_level=max_templates_per_level)

    summary: Dict[str, Dict[str, float | List[str]]] = {
        "baseline": {"templates": [], "identity_threshold": None},
    }

    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _invoke_predict(
        target_fasta,
        baseline_dir,
        cache,
        checkpoint,
        model_size,
        token_per_batch,
        num_recycling,
        compile,
        kernels,
        template_override=None,
    )

    for level in sorted(buckets.keys(), reverse=True):
        templates_for_level = buckets[level]
        if not templates_for_level:
            continue

        dist = build_distogram_from_templates(len(record.seq), templates_for_level)
        override_path = _write_template_payload(output_dir, record.id, dist)

        label = f"id_{int(level * 100)}"
        run_dir = output_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)

        _invoke_predict(
            target_fasta,
            run_dir,
            cache,
            checkpoint,
            model_size,
            token_per_batch,
            num_recycling,
            compile,
            kernels,
            template_override=override_path,
        )

        summary[label] = {
            "identity_threshold": level,
            "templates": [template.name for template in templates_for_level],
        }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    click.echo(f"Completed quick test. Summary written to {summary_path}")


if __name__ == "__main__":
    main()

