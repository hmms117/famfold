# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime
from multiprocessing import cpu_count
from typing import Mapping, Optional, Sequence, Any

import numpy as np
import torch
import datetime

from minifold.data import parsers, mmcif_parsing
from minifold.utils import residue_constants, protein


FeatureDict = Mapping[str, np.ndarray]


def to_date(s: str):
    return datetime.datetime(
        year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10])
    )


def empty_template_feats(n_res) -> FeatureDict:
    return {
        "template_aatype": np.zeros((0, n_res)).astype(np.int64),
        "template_all_atom_positions":
            np.zeros((0, n_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, n_res, 37)).astype(np.float32),
    }


def make_template_features(
        input_sequence: str,
        hits: Sequence[Any],
        template_featurizer: Any,
        query_pdb_code: Optional[str] = None,
        query_release_date: Optional[str] = None,
) -> FeatureDict:
    hits_cat = sum(hits.values(), [])
    if (len(hits_cat) == 0 or template_featurizer is None):
        template_features = empty_template_feats(len(input_sequence))
    else:
        templates_result = template_featurizer.get_templates(
            query_sequence=input_sequence,
            query_pdb_code=query_pdb_code,
            query_release_date=query_release_date,
            hits=hits_cat,
        )
        template_features = templates_result.features

        # The template featurizer doesn't format empty template features
        # properly. This is a quick fix.
        if (template_features["template_aatype"].shape[0] == 0):
            template_features = empty_template_feats(len(input_sequence))

    return template_features


def make_sequence_features(
        sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=object
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=object
    )
    return features


def make_mmcif_features(
        mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask, all_atom_loss_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_loss_mask
    mmcif_feats["all_atom_mask_true"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=object
    )

    mmcif_feats["is_distillation"] = np.array(1., dtype=np.float32)

    return mmcif_feats


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]]
        for i in range(len(aatype))
    ])


def make_protein_features(
        protein_object: protein.Protein,
        description: str,
        _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)

    return pdb_feats


def make_pdb_features(
        protein_object: protein.Protein,
        description: str,
        is_distillation: bool = True,
        confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if (is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def make_msa_features(
        msas: Sequence[Sequence[str]],
        deletion_matrices: Sequence[parsers.DeletionMatrix],
) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f"MSA {msa_index} must contain at least one sequence."
            )
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
            )
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array(
        [num_alignments] * num_res, dtype=np.int32
    )
    return features


# Generate 1-sequence MSA features having only the input sequence
def make_dummy_msa_feats(input_sequence):
    msas = [[input_sequence]]
    deletion_matrices = [[[0 for _ in input_sequence]]]
    msa_features = make_msa_features(
        msas=msas,
        deletion_matrices=deletion_matrices,
    )

    return msa_features


class DataPipeline:
    """Assembles input features."""

    def __init__(
            self,
            template_featurizer,
    ):
        self.template_featurizer = template_featurizer

    def _parse_msa_data(
            self,
            alignment_dir: str,
            alignment_index: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        msa_data = {}
        for f in os.listdir(alignment_dir):
            path = os.path.join(alignment_dir, f)
            ext = os.path.splitext(f)[-1]

            if (ext == ".a3m"):
                with open(path, "r") as fp:
                    msa, deletion_matrix = parsers.parse_a3m(fp.read())
                data = {"msa": msa, "deletion_matrix": deletion_matrix}
            elif (ext == ".sto"):
                with open(path, "r") as fp:
                    msa, deletion_matrix, _ = parsers.parse_stockholm(
                        fp.read()
                    )
                data = {"msa": msa, "deletion_matrix": deletion_matrix}
            else:
                continue

            msa_data[f] = data

        return msa_data

    def _parse_template_hits(
            self,
            alignment_dir: str,
            alignment_index: Optional[Any] = None
    ) -> Mapping[str, Any]:
        all_hits = {}
        if alignment_index is not None:
            fp = open(os.path.join(alignment_dir, alignment_index["db"]), 'rb')

            def read_template(start, size):
                fp.seek(start)
                return fp.read(size).decode("utf-8")

            for (name, start, size) in alignment_index["files"]:
                ext = os.path.splitext(name)[-1]

                if (ext == ".hhr"):
                    hits = parsers.parse_hhr(read_template(start, size))
                    all_hits[name] = hits

            fp.close()
        else:
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if (ext == ".hhr"):
                    with open(path, "r") as fp:
                        hits = parsers.parse_hhr(fp.read())
                    all_hits[f] = hits

        return all_hits

    def _get_msas(self,
                  alignment_dir: str,
                  input_sequence: Optional[str] = None,
                  alignment_index: Optional[str] = None,
                  ):
        msa_data = {}
        if (len(msa_data) == 0):
            if (input_sequence is None):
                raise ValueError(
                    """
                    If the alignment dir contains no MSAs, an input sequence 
                    must be provided.
                    """
                )
            msa_data["dummy"] = {
                "msa": [input_sequence],
                "deletion_matrix": [[0 for _ in input_sequence]],
            }

        msas, deletion_matrices = zip(*[
            (v["msa"], v["deletion_matrix"]) for v in msa_data.values()
        ])

        return msas, deletion_matrices

    def _process_msa_feats(
            self,
            alignment_dir: str,
            input_sequence: Optional[str] = None,
            alignment_index: Optional[str] = None
    ) -> Mapping[str, Any]:
        msas, deletion_matrices = self._get_msas(
            alignment_dir, input_sequence, alignment_index
        )
        msa_features = make_msa_features(
            msas=msas,
            deletion_matrices=deletion_matrices,
        )

        return msa_features

    # Load and process sequence embedding features
    def _process_seqemb_features(self,
                                 alignment_dir: str,
                                 ) -> Mapping[str, Any]:
        seqemb_features = {}
        for f in os.listdir(alignment_dir):
            path = os.path.join(alignment_dir, f)
            ext = os.path.splitext(f)[-1]

            if (ext == ".pt"):
                # Load embedding file
                seqemb_data = torch.load(path)
                seqemb_features["seq_embedding"] = seqemb_data["representations"][33]

        return seqemb_features

    def process_fasta(
            self,
            fasta_path: str,
            alignment_dir: str,
            alignment_index: Optional[str] = None,
            seqemb_mode: bool = False,
    ) -> FeatureDict:
        """Assembles features for a single sequence in a FASTA file"""
        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f"More than one input sequence found in {fasta_path}."
            )
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        sequence_embedding_features = {}
        # If using seqemb mode, generate a dummy MSA features using just the sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {
            **sequence_features,
            **msa_features,
        }

    def process_mmcif(
            self,
            mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
            alignment_dir: str,
            chain_id: Optional[str] = None,
            alignment_index: Optional[str] = None,
            seqemb_mode: bool = False,
    ) -> FeatureDict:
        """
            Assembles features for a specific chain in an mmCIF object.

            If chain_id is None, it is assumed that there is only one chain
            in the object. Otherwise, a ValueError is thrown.
        """
        if chain_id is None:
            chains = mmcif.structure.get_chains()
            chain = next(chains, None)
            if chain is None:
                raise ValueError("No chains in mmCIF file")
            chain_id = chain.id

        mmcif_feats = make_mmcif_features(mmcif, chain_id)

        input_sequence = mmcif.chain_to_seqres[chain_id]
        """
        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
            query_release_date=to_date(mmcif.header["release_date"])
        )
        """
        # If using seqemb mode, generate a dummy MSA features using just the sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)
        return {**mmcif_feats, **msa_features}

    def process_pdb(
            self,
            pdb_path: str,
            alignment_dir: str,
            is_distillation: bool = True,
            chain_id: Optional[str] = None,
            _structure_index: Optional[str] = None,
            alignment_index: Optional[str] = None,
            seqemb_mode: bool = False,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file.
        """
        if (_structure_index is not None):
            db_dir = os.path.dirname(pdb_path)
            db = _structure_index["db"]
            db_path = os.path.join(db_dir, db)
            fp = open(db_path, "rb")
            _, offset, length = _structure_index["files"][0]
            fp.seek(offset)
            pdb_str = fp.read(length).decode("utf-8")
            fp.close()
        else:
            with open(pdb_path, 'r') as f:
                pdb_str = f.read()

        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features(
            protein_object,
            description,
            is_distillation=is_distillation
        )

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        sequence_embedding_features = {}
        # If in sequence embedding mode, generate dummy MSA features using just the input sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
            sequence_embedding_features = self._process_seqemb_features(alignment_dir)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {**pdb_feats, **template_features, **msa_features, **sequence_embedding_features}

    def process_core(
            self,
            core_path: str,
            alignment_dir: str,
            alignment_index: Optional[str] = None,
            seqemb_mode: bool = False,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a ProteinNet .core file.
        """
        with open(core_path, 'r') as f:
            core_str = f.read()

        protein_object = protein.from_proteinnet_string(core_str)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(core_path))[0].upper()
        core_feats = make_protein_features(protein_object, description)

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        sequence_embedding_features = {}
        # If in sequence embedding mode, generate dummy MSA features using just the input sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
            sequence_embedding_features = self._process_seqemb_features(alignment_dir)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence)

        return {**core_feats, **template_features, **msa_features, **sequence_embedding_features}
