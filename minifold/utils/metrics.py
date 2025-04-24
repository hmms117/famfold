import torch
import numpy as np

from minifold.utils import residue_constants
from minifold.utils.tensor_utils import permute_final_dims


def lddt(preds, target, mask, cutoff=15.0, per_residue=False):
    """Compute lDDT score."""
    mask_ = mask
    # Compute true and predicted distance matrices.
    dmat_true = torch.cdist(target, target)
    dmat_predicted = torch.cdist(preds, preds)

    # Compute mask over distances
    mask = mask[:, None, :] * mask[:, :, None]
    mask = mask * (1 - torch.eye(mask.shape[1], device=mask.device))[None, :, :]
    dists_to_score = (dmat_true < cutoff).float() * mask

    # Shift unscored distances to be far away.
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=reduce_axes))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=reduce_axes))

    if per_residue:
        return score * mask_
    else:
        return score.mean()


def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_residue=False):
    mask_ = mask
    """Compute lDDT score."""
    # Compute mask over distances
    mask = mask[:, None, :] * mask[:, :, None]
    mask = mask * (1 - torch.eye(mask.shape[1], device=mask.device))[None, :, :]
    dists_to_score = (dmat_true < cutoff).float() * mask

    # Shift unscored distances to be far away.
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=reduce_axes))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=reduce_axes))
    if per_residue:
        return score * mask_
    else:
        return score.mean()


def lddt_np(
    predicted_points, true_points, true_points_mask, cutoff=15.0, per_residue=False
):
    """Measure (approximate) lDDT for a batch of coordinates.
    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).
    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.
    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.
    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
        cutoff: Maximum distance for a pair of points to be included
        per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.
    Returns:
        An (approximate, see above) lDDT score in the range 0-1.
    """
    assert len(predicted_points.shape) == 3
    assert predicted_points.shape[-1] == 3
    assert true_points_mask.shape[-1] == 1
    assert len(true_points_mask.shape) == 3

    # Compute true and predicted distance matrices.
    dmat_true = np.sqrt(
        1e-10
        + np.sum((true_points[:, :, None] - true_points[:, None, :]) ** 2, axis=-1)
    )

    dmat_predicted = np.sqrt(
        1e-10
        + np.sum(
            (predicted_points[:, :, None] - predicted_points[:, None, :]) ** 2, axis=-1
        )
    )

    dists_to_score = (
        (dmat_true < cutoff).astype(np.float32)
        * true_points_mask
        * np.transpose(true_points_mask, [0, 2, 1])
        * (1.0 - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = np.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * (
        (dist_l1 < 0.5).astype(np.float32)
        + (dist_l1 < 1.0).astype(np.float32)
        + (dist_l1 < 2.0).astype(np.float32)
        + (dist_l1 < 4.0).astype(np.float32)
    )

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))
    return score


def lddt_np_dist(
    dmat_predicted, dmat_true, true_points_mask, cutoff=15.0, per_residue=False
):
    """Measure (approximate) lDDT for a batch of coordinates.
    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).
    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.
    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.
    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
        cutoff: Maximum distance for a pair of points to be included
        per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.
    Returns:
        An (approximate, see above) lDDT score in the range 0-1.
    """
    dists_to_score = (
        (dmat_true < cutoff).astype(np.float32)
        * true_points_mask
        * np.transpose(true_points_mask, [0, 2, 1])
        * (1.0 - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = np.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * (
        (dist_l1 < 0.5).astype(np.float32)
        + (dist_l1 < 1.0).astype(np.float32)
        + (dist_l1 < 2.0).astype(np.float32)
        + (dist_l1 < 4.0).astype(np.float32)
    )

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))
    return score


def lddt_of(
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
        per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
            (dmat_true < cutoff)
            * all_atom_mask
            * permute_final_dims(all_atom_mask, (1, 0))
            * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
            (dist_l1 < 0.5).type(dist_l1.dtype)
            + (dist_l1 < 1.0).type(dist_l1.dtype)
            + (dist_l1 < 2.0).type(dist_l1.dtype)
            + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_of_ca(
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
        per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos: (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )