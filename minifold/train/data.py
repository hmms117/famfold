from typing import List, Any, Optional
import os

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from minifold.utils.esm import load_model_and_alphabet

from minifold.data.of_data import get_data
from minifold.data.config import model_config
from minifold.utils.residue_constants import restype_order_with_x_inverse


def process(
    path: str,
    max_length: int,
    alphabet: Any,
    generator: Any,
    config,
):
    # Get Openfold batch
    open_fold_batch = get_data(
        path.split("/")[-1].replace(".cif", ""), path, "train", config
    )

    # Get sequence
    l = open_fold_batch["seq_length"]
    of_seq = "".join(
        [restype_order_with_x_inverse[x.item()] for x in open_fold_batch["aatype"]]
    )[:l]

    # Encode for minifold
    encoded_seq = alphabet.encode(of_seq)
    pad_idx = alphabet.padding_idx

    pad_len = max_length - l.item()
    if pad_len > 0:
        encoded_seq = np.pad(encoded_seq, (0, pad_len), constant_values=pad_idx)

    encoded_seq = torch.tensor(encoded_seq, dtype=torch.long)

    batch = {
        "seq": encoded_seq,
        "coords": open_fold_batch["all_atom_positions"][:, 0:3, :, 0],
        "mask": open_fold_batch["seq_mask"][:, 0].bool(),
        "batch_of": open_fold_batch,
    }

    return batch


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        alphabet: Any,
        files: List[str],
        max_length: int = 256,
        samples_per_epoch: int = 1000000,
        config=None,
    ):
        super().__init__()
        self.alphabet = alphabet
        self.files = files
        self.max_length = max_length
        self.length = samples_per_epoch
        self.config = config

    def __getitem__(self, idx):
        # Pick random file
        file_idx = np.random.randint(len(self.files))
        file_path = self.files[file_idx]

        # Process file
        try:
            return process(
                path=file_path,
                max_length=self.max_length,
                alphabet=self.alphabet,
                generator=np.random,
                config=self.config,
            )
        except Exception as e:
            print("Error", e)
            # If there are some erronous files in the database
            # We fallback to the first file for safety
            file_idx = 0
            file_path = self.files[file_idx]
            return process(
                path=file_path,
                max_length=self.max_length,
                alphabet=self.alphabet,
                generator=np.random,
                config=self.config,
            )

    def __len__(self):
        return self.length


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, alphabet: Any, files: str, max_length: int = 256, config=None):
        super().__init__()
        self.alphabet = alphabet
        self.files = files
        self.max_length = max_length
        self.random = np.random.RandomState(42)
        self.config = config

    def __getitem__(self, idx):
        # Pick file
        file_path = self.files[idx]

        # Process file
        try:
            return process(
                path=file_path,
                max_length=self.max_length,
                alphabet=self.alphabet,
                generator=self.random,
                config=self.config,
            )
        except Exception as e:
            print("Error", e)
            # If there are some erronous files in the database
            # We fallback to the first file for safety
            file_idx = 0
            file_path = self.files[file_idx]
            return process(
                path=file_path,
                max_length=self.max_length,
                alphabet=self.alphabet,
                generator=self.random,
                config=self.config,
            )

    def __len__(self):
        return len(self.files)


class MiniFoldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        esm_model_name="esm2_t36_3B_UR50D",
        esm_cache_path: Optional[str] = None,
        num_workers: int = 0,
        samples_per_epoch: int = 1000000,
        max_length: int = 256,
        seed: int = 42,
        overfit: bool = False,
        ignore: Optional[str] = None,
    ):
        super().__init__()
        # Set cache path
        if esm_cache_path is not None:
            torch.hub.set_dir(esm_cache_path)

        # Save params
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Find all folders
        folders = sorted(list(os.listdir(data_dir)))
        folders = [os.path.join(data_dir, folder) for folder in folders]
        folders = [folder for folder in folders if os.path.isdir(folder)]

        # Load all files
        files = []
        for folder in folders:
            names = sorted(list(os.listdir(folder)))
            names = [os.path.join(folder, name) for name in names]
            names = [name for name in names if name.endswith(".cif")]
            files.extend(names)

        # Split into train and validation
        train, val = train_test_split(files, test_size=10000, random_state=seed)

        # Load ignore file
        if ignore is not None:
            ignore_files = set(pd.read_csv(ignore)["id_2"])
            train = [f for f in train if os.path.basename(f) not in ignore_files]
            print(f"Ignore: {len(ignore_files)}")

        if overfit:
            train = train[:10]
            val = train[:10]
            samples_per_epoch = 1000

        print(f"Train: {len(train)}")
        print(f"Val: {len(val)}")

        # Load alphabet
        _, alphabet = load_model_and_alphabet(esm_model_name)

        # Load of config
        config_openfold = model_config(
            "initial_training",
            train=True,
            low_prec=False,
            long_sequence_inference=False,
        )

        # Create datasets
        self.train_dataset = TrainingDataset(
            alphabet=alphabet,
            files=train,
            max_length=max_length,
            samples_per_epoch=samples_per_epoch,
            config=config_openfold.data,
        )
        self.val_dataset = ValidationDataset(
            alphabet=alphabet,
            files=val,
            max_length=max_length,
            config=config_openfold.data,
        )

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
        return dl

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
        return dl
