from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import MeanMetric

from minifold.data.config import model_config
from minifold.model.model import MiniFoldModel
from minifold.train.loss import AlphaFoldLoss
from minifold.utils.metrics import lddt_dist, lddt_of_ca
from minifold.utils.tensor_utils import tensor_tree_map


class MiniFold(pl.LightningModule):
    def __init__(
        self,
        esm_model_name="esm2_t33_650M_UR50D",
        base_lr: float = 1e-4,
        lm_lr: float = 3e-5,
        struct_lr: float = 1e-4,
        num_blocks: int = 8,
        pretrained: Optional[str] = None,
        num_recycling: int = 0,
        max_dist: float = 25.0,
        no_bins: int = 64,
        compile: bool = True,
        debug: bool = False,
        use_structure_module: bool = False,
        num_structure_blocks: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base_lr = base_lr
        self.lm_lr = lm_lr
        self.struct_lr = struct_lr
        self.esm_model_name = esm_model_name
        self.no_bins = no_bins
        self.num_recycling = num_recycling
        self.random_state = np.random.RandomState(42)
        self.max_dist = max_dist
        self.debug = debug
        self.use_structure_module = use_structure_module
        self.config_of = model_config(
            "initial_training",
            train=True,
            low_prec=False,
            long_sequence_inference=False,
        )

        # Creat model
        model = MiniFoldModel(
            esm_model_name=esm_model_name,
            num_blocks=num_blocks,
            no_bins=self.no_bins,
            use_structure_module=use_structure_module,
            config_of=self.config_of,
            num_structure_blocks=num_structure_blocks,
        )
        self.model = model
        self.loss_of = AlphaFoldLoss(self.config_of.loss)

        # Compute distogram
        boundaries = torch.linspace(2, max_dist, self.no_bins - 1)
        lower = torch.tensor([1.0])
        upper = torch.tensor([max_dist + 5.0])
        exp_boundaries = torch.cat((lower, boundaries, upper))
        mid_points = (exp_boundaries[:-1] + exp_boundaries[1:]) / 2
        self.register_buffer("boundaries", boundaries)
        self.register_buffer("mid_points", mid_points)

        # Load pretrained weights
        if pretrained is not None:
            ckpt = torch.load(pretrained, map_location="cpu")
            state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()
            }
            state_dict = {k: v for k, v in state_dict.items() if "boundaries" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "mid_points" not in k}
            self.load_state_dict(state_dict, strict=False)

        # Compile folding block
        if compile:
            self.model.fold.miniformer = torch.compile(
                self.model.fold.miniformer,
                dynamic=False,
                fullgraph=True,
            )

        # Metric accumulator
        self.disto_metric = MeanMetric()
        self.struct_metric = MeanMetric()

    def forward(self, batch, num_recycling=0):
        return self.model(batch, num_recycling)

    def training_step(self, batch, batch_idx):
        # Pick random reycling
        recyling = self.random_state.randint(0, self.num_recycling + 1)

        # Compute predictions
        r_dict = self(batch, num_recycling=recyling)
        preds = r_dict["preds"]

        # Get ground truth distogram
        coords = batch["coords"]
        coords = coords[:, :, 1, :]
        dists = torch.cdist(coords, coords)

        labels = (
            torch.nn.functional.one_hot(
                (dists.unsqueeze(-1) > self.boundaries).sum(dim=-1), self.no_bins
            )
        ).to(preds)
        errors = -1 * torch.sum(
            labels * torch.nn.functional.log_softmax(preds, dim=-1),
            dim=-1,
        )

        mask = batch["mask"]
        square_mask = mask[:, None] * mask[:, :, None]
        square_mask = square_mask * (1 - torch.eye(dists.shape[1])[None]).to(dists)

        # FP16-friendly sum
        denom = 1e-5 + torch.sum(square_mask, dim=(-1, -2))
        mean = errors * square_mask
        mean = torch.sum(mean, dim=-1)
        mean = mean / denom[..., None]
        mean = torch.sum(mean, dim=-1)
        disto_loss = torch.mean(mean)

        if self.use_structure_module:
            # Remove the recycling dimension
            batch_of = batch["batch_of"]
            batch_of = tensor_tree_map(lambda t: t[..., -1], batch_of)

            # Compute loss
            loss_of, loss_breakdown = self.loss_of(
                r_dict, batch_of, _return_breakdown=True
            )

            # Compute total loss
            total_loss = loss_of * 0.2 + disto_loss * 0.8

            for loss_name, indiv_loss in loss_breakdown.items():
                self.log(f"train/of_{loss_name}", indiv_loss, on_step=True)
        else:
            total_loss = disto_loss

        self.log("train/loss", total_loss, prog_bar=True)
        if batch_idx % 100 == 0:
            self.log("train/grad_norm", self.gradient_norm, prog_bar=False)
            self.log("train/param_norm", self.parameter_norm, prog_bar=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Get prediction
        r_dict = self(batch, num_recycling=self.num_recycling)
        preds = r_dict["preds"]

        # Compute argmax distances
        pred_softmax = torch.softmax(preds, dim=-1)
        pred_softmax = pred_softmax.argmax(dim=-1)
        pred_softmax = torch.nn.functional.one_hot(
            pred_softmax, num_classes=preds.shape[-1]
        )
        dists = (pred_softmax * self.mid_points).sum(dim=-1)

        # Apply self masking
        dists = dists * (1 - torch.eye(dists.shape[1])[None]).to(dists)

        # Get C-a ground truth
        true_coords = batch["coords"]
        true_coords = true_coords[:, :, 1, :]
        true_dist = torch.cdist(true_coords, true_coords)

        score = lddt_dist(dists, true_dist, batch["mask"].to(dists))
        self.disto_metric.update(score)

        if self.use_structure_module:
            # Remove the recycling dimension
            batch_of = batch["batch_of"]
            batch_of = tensor_tree_map(lambda t: t[..., -1], batch_of)

            gt_coords = batch_of["all_atom_positions"]
            pred_coords = r_dict["final_atom_positions"]
            all_atom_mask = batch_of["all_atom_mask_true"]
            lddt_ca_score = lddt_of_ca(
                pred_coords,
                gt_coords,
                all_atom_mask,
                eps=self.config_of.globals.eps,
                per_residue=False,
            )
            self.struct_metric.update(lddt_ca_score)

    def on_validation_epoch_end(self):
        disto_lddt = self.disto_metric.compute()
        struct_lddt = self.struct_metric.compute()
        self.log("val/disto_lddt", disto_lddt, prog_bar=True)
        self.log("val/lddt", struct_lddt, prog_bar=False)
        self.disto_metric.reset()
        self.struct_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [
                        p
                        for name, p in self.named_parameters()
                        if p.requires_grad
                        and (
                            ("lm" not in name)
                            and ("structure_module" not in name)
                            and ("aux_heads" not in name)
                            and ("sz_project" not in name)
                        )
                    ],
                    "lr": self.base_lr,
                },
                {
                    "params": [
                        p
                        for name, p in self.named_parameters()
                        if p.requires_grad
                        and (
                            ("structure_module" in name)
                            or ("aux_heads" in name)
                            or ("sz_project" in name)
                        )
                    ],
                    "lr": self.struct_lr,
                },
                {
                    "params": [
                        p
                        for name, p in self.named_parameters()
                        if p.requires_grad and ("lm" in name)
                    ],
                    "lr": self.lm_lr,
                },
            ],
            lr=self.base_lr,
        )
        return optimizer

    @property
    def gradient_norm(self) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(
            lambda p: p.requires_grad and p.grad is not None, self.parameters()
        )
        norm = (
            torch.tensor([param.grad.norm(p=2) ** 2 for param in parameters])
            .sum()
            .sqrt()
        )
        return norm

    @property
    def parameter_norm(self) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        norm = torch.tensor([param.norm(p=2) ** 2 for param in parameters]).sum().sqrt()
        return norm

    def on_save_checkpoint(self, checkpoint):
        # Don't save frozen weights from the lm
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if ("lm" not in k)
            or ("layers.34" in k)
            or ("layers.35" in k)
            or ("emb_layer_norm_after" in k)
        }
