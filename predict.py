import os
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import click
import torch

import torch.nn.functional as F
import numpy as np
from Bio import SeqIO
from minifold.utils.esm import load_model_and_alphabet
from tqdm import tqdm

from minifold.model.model import MiniFoldModel
from minifold.utils.residue_constants import restype_order_with_x
from minifold.utils.protein import Protein, to_pdb
from minifold.data.config import model_config
from minifold.data.of_data import of_inference
from minifold.utils.residue_constants import restype_order_with_x_inverse


MODEL_URL_48L = (
    "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_48L_final.ckpt"
)

MODEL_URL_12L = (
    "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_12L_final.ckpt"
)


def download(cache: Path, model_size: str) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    model = cache / f"minifold_{model_size}.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        url = MODEL_URL_48L if model_size == "48L" else MODEL_URL_12L
        urllib.request.urlretrieve(url, str(model))  # noqa: S310


def output_to_pdb(seq, coords, mask, plddt):
    """Returns the pbd (file) string from the model given the model output."""
    mapping = restype_order_with_x
    af_seq = np.array([mapping[res] for res in seq])
    plddt = plddt[:, None].repeat(coords.shape[1], axis=1)
    pred = Protein(
        aatype=af_seq,
        atom_positions=coords,
        atom_mask=mask,
        residue_index=1 + np.arange(len(seq)),
        b_factors=plddt,
    )
    return to_pdb(pred)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_model(checkpoint, device, compile=False, kernels=False):
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt["hyper_parameters"]

    if kernels:
        torch._dynamo.config.cache_size_limit = 64

    # Initialize minifold model object
    config_of = model_config(
        "initial_training",
        train=False,
        low_prec=False,
        long_sequence_inference=False,
    )
    model = MiniFoldModel(
        esm_model_name=hparams["esm_model_name"],
        num_blocks=hparams["num_blocks"],
        no_bins=hparams["no_bins"],
        config_of=config_of,
        use_structure_module=True,
        kernels=kernels,
    )

    # Initialize alphabet
    _, alphabet = load_model_and_alphabet(hparams["esm_model_name"])

    # Load pretrained weights
    state_dict = ckpt["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if "boundaries" not in k}
    state_dict = {k: v for k, v in state_dict.items() if "mid_points" not in k}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Compile folding block
    if compile:
        model.fold.miniformer = torch.compile(
            model.fold.miniformer,
            dynamic=True,
            fullgraph=True,
        )

    # Move model to device
    model = model.to(device)
    model.eval()

    return alphabet, model


def create_batches(fasta, token_per_batch):
    # Load sequences and sort by length
    seqs = list(SeqIO.parse(fasta, "fasta"))
    seqs = sorted(seqs, key=lambda x: len(str(x.seq)))

    # Create batches
    batches = []
    while len(seqs) > 0:
        seq = seqs.pop(0)
        batch = [seq]

        while True:
            if len(seqs) == 0:
                break

            next_seq = seqs[0]
            next_batch_size = (len(batch) + 1) * len(str(next_seq.seq))

            if next_batch_size > token_per_batch:
                break

            seq = seqs.pop(0)
            batch.append(seq)

        batches.append(batch)
    return batches


def prepare_input(seq, config, alphabet):
    # Get Openfold batch
    open_fold_batch = of_inference(
        seq,
        "predict",
        config,
    )

    # Get sequence
    of_seq = "".join(
        [restype_order_with_x_inverse[x.item()] for x in open_fold_batch["aatype"]]
    )[: open_fold_batch["seq_length"]]

    # Encode for minifold
    encoded_seq = alphabet.encode(of_seq)
    encoded_seq = torch.tensor(encoded_seq, dtype=torch.long)

    # Prepare batch
    mask = open_fold_batch["seq_mask"][:, 0].bool()

    # Keep only relevant keys
    relevant = {"aatype", "seq_mask", "residx_atom37_to_atom14", "atom37_atom_exists"}
    open_fold_batch = {k: v for k, v in open_fold_batch.items() if k in relevant}
    return encoded_seq, mask, open_fold_batch


@click.command()
@click.argument("fasta", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./minifold_predictions",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.minifold, or $MINIFOLD_CACHE if set.",
    default="./minifold_cache",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Minifold model by default.",
    default=None,
)
@click.option(
    "--token_per_batch",
    type=int,
    default=2048,
    help="The number of tokens per batch. Default is 2048.",
)
@click.option(
    "--compile",
    is_flag=True,
    help="Whether to compile the model. Default is False.",
)
@click.option(
    "--model_size",
    type=click.Choice(["48L", "12L"]),
    default="48L",
    help="The size of the model to use. Default is 48L.",
)
@click.option(
    "--kernels",
    is_flag=True,
    help="Whether to use kernels. Default is False.",
)
@click.option(
    "--num_recycling",
    type=int,
    default=3,
    help="Number of recycling iterations for the structure module (default: 3).",
)
@click.option(
    "--template_overrides",
    type=click.Path(exists=True),
    default=None,
    help="Path to a torch.save() file mapping FASTA record identifiers to template distograms.",
)
def predict(
    fasta: str,
    out_dir: str = "./minifold_predictions",
    cache: str = "./minifold_cache",
    checkpoint: Optional[str] = None,
    token_per_batch: int = 2048,
    compile: bool = False,
    model_size: str = "48L",
    kernels: bool = False,
    num_recycling: int = 3,
    template_overrides: Optional[str] = None,
) -> None:
    """Run predictions with Minifold."""
    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    fasta = Path(fasta).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"minifold_results_{fasta.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache, model_size)

    # Detect device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    if checkpoint is None:
        checkpoint = cache / f"minifold_{model_size}.ckpt"

    # Set hub cache
    torch.hub.set_dir(cache)

    # Disable gradient computation
    torch.set_grad_enabled(False)

    # Load checkpoint
    print("Load model...")
    alphabet, model = create_model(checkpoint, device, compile)

    # Create batches
    config = model_config(
        "initial_training",
        train=False,
        low_prec=False,
        long_sequence_inference=False,
    )
    config = config.data

    # Create batches
    print("Load sequences...")
    batches = create_batches(fasta, token_per_batch)

    # Load optional template overrides
    overrides: Dict[str, Dict[str, torch.Tensor]] = {}
    if template_overrides is not None:
        overrides_raw = torch.load(template_overrides, map_location="cpu")
        if isinstance(overrides_raw, dict):
            overrides = {
                str(key): {
                    "distogram": torch.as_tensor(value.get("distogram", value)),
                }
                for key, value in overrides_raw.items()
            }
        else:
            raise TypeError("Template overrides file must contain a dictionary mapping sequence identifiers to tensors.")

    # Predict
    print("Launching predictions...")
    for batch in tqdm(batches):
        # Prepare batch
        feats = [prepare_input(str(t.seq), config, alphabet) for t in batch]

        # Pad everything
        max_len = max(len(seq) for seq, _, _ in feats)

        if kernels:
            # Round up to the next multiple of 128
            # in case kernels are being used
            max_len = (max_len + 127) // 128 * 128

        # Pad sequences
        seq = torch.stack(
            [F.pad(seq, (0, max_len - len(seq)), value=20) for seq, _, _ in feats]
        )
        mask = torch.stack(
            [F.pad(mask, (0, max_len - len(mask)), value=0) for _, mask, _ in feats]
        )
        batch_of = {}
        for _, _, feats_of in feats:
            for k, v in feats_of.items():
                batch_of.setdefault(k, []).append(v)

        for k, v in batch_of.items():
            batch_of[k] = torch.stack(
                [
                    F.pad(
                        item,
                        [0] * 2 * (len(item.shape) - 1) + [0, max_len - item.shape[0]],
                        value=0,
                    )
                    for item in v
                ]
            )

        # Move to device
        model_batch = {
            "seq": seq.to(device),
            "mask": mask.to(device),
            "batch_of": {k: v.to(device) for k, v in batch_of.items()},
        }

        if overrides:
            template_batch = torch.zeros(
                len(batch),
                max_len,
                max_len,
                model.fold.disto_bins,
                dtype=torch.float32,
            )
            any_template = False
            for idx, record in enumerate(batch):
                entry = overrides.get(str(record.id))
                if entry is None:
                    continue
                dist_tensor = entry.get("distogram")
                if dist_tensor is None:
                    continue
                dist_tensor = torch.as_tensor(dist_tensor, dtype=torch.float32)
                if dist_tensor.ndim != 3:
                    raise ValueError("Template distogram must be a rank-3 tensor (L, L, bins).")
                template_len = dist_tensor.shape[0]
                if dist_tensor.shape[1] != template_len:
                    raise ValueError("Template distogram must have square pair dimensions.")
                if dist_tensor.shape[2] != model.fold.disto_bins:
                    raise ValueError(
                        f"Template distogram expects {model.fold.disto_bins} bins but received {dist_tensor.shape[2]}."
                    )
                if template_len > max_len:
                    raise ValueError(
                        f"Template distogram length {template_len} exceeds padded batch length {max_len}."
                    )
                padded = F.pad(
                    dist_tensor,
                    (0, 0, 0, max_len - template_len, 0, max_len - template_len),
                    value=0.0,
                )
                template_batch[idx] = padded
                any_template = True

            if any_template:
                model_batch["template_distogram"] = template_batch.to(device)

        # Run predictions
        try:
            autocast_device = "cuda" if device.type == "cuda" else ("cpu" if device.type == "cpu" else "mps")
            with torch.autocast(autocast_device, dtype=torch.bfloat16):
                out = model(model_batch, num_recycling=num_recycling)
                out_pos = out.pop("final_atom_positions")
                out_mask = out.pop("final_atom_mask")
                plddt = out.pop("plddt")
                del out

            out_pos = out_pos.float().cpu().numpy()
            out_mask = out_mask.float().cpu().numpy()
            plddt = plddt.float().cpu().numpy()

            for idx, t in enumerate(batch):
                # Set output path
                path_out = out_dir / f"{str(t.id)}.pdb"

                # unpad and save
                out_pos_i = out_pos[idx, : len(t.seq)]
                out_mask_i = out_mask[idx, : len(t.seq)]
                plddt_i = plddt[idx, : len(t.seq)]

                pdb_string = output_to_pdb(str(t.seq), out_pos_i, out_mask_i, plddt_i)
                with open(path_out, "w") as f:
                    f.write(pdb_string)

        # Skip if prediction failed
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Failed to predict batch: {e}")
            continue


if __name__ == "__main__":
    predict()
