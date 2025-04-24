import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from minifold.utils import init
from minifold.model.kernels.gating import gating_kernel
from minifold.model.kernels.mlp import mlp_kernel


def mlp(x, w1, w2, b1, b2, wn, bn):
    """Perform a two-layer MLP with a residual connection."""
    x = F.layer_norm(x, [x.shape[-1]], wn, bn, eps=1e-5)
    x = F.linear(x, w1, b1)
    x = F.relu(x)
    x = F.linear(x, w2, b2)
    return x


def triangular(
    x,
    mask,
    pi_w,
    gi_w,
    pi_b,
    gi_b,
    po_w,
    go_w,
    po_b,
    go_b,
    ni_w,
    ni_b,
    no_w,
    no_b,
):
    """Perform a triangular update"""
    # Input gating: D -> D
    x = F.layer_norm(x, [x.shape[-1]], ni_w, ni_b, eps=1e-5)
    x = F.linear(x, pi_w, pi_b) * F.linear(x, gi_w, gi_b).sigmoid()

    # Apply mask
    x = x * mask.unsqueeze(-1)

    # Split input and cast to float
    with torch.autocast("cuda", enabled=False):
        a1, b1, a2, b2 = torch.chunk(x.float(), 4, dim=-1)

        # Triangular projection
        x1 = torch.einsum("bikd,bjkd->bijd", a1, b1)
        x2 = torch.einsum("bkid,bkjd->bijd", a2, b2)

        # Merge outputs
        x = torch.cat([x1, x2], dim=-1).to(x.dtype)

    # Output gating: D / 2 -> D
    x = F.layer_norm(x, [x.shape[-1]], no_w, no_b, eps=1e-5)
    x = F.linear(x, po_w, po_b) * F.linear(x, go_w, go_b).sigmoid()

    return x


def mlp_kernel_func(x, w1, w2, b1, b2, wn, bn):
    """Perform a two-layer MLP with a residual connection."""
    w1 = w1.t().contiguous()
    w2 = w2.t().contiguous()
    return mlp_kernel(x, w1, w2, b1, b2, wn, bn)


def triangular_kernel_func(
    x,
    mask,
    pi_w,
    gi_w,
    pi_b,
    gi_b,
    po_w,
    go_w,
    po_b,
    go_b,
    ni_w,
    ni_b,
    no_w,
    no_b,
):
    """Perform a triangular update"""
    # Tranpose weights
    gi_w = gi_w.t().contiguous()
    pi_w = pi_w.t().contiguous()
    go_w = go_w.t().contiguous()
    po_w = po_w.t().contiguous()

    # Input gating: D -> D
    x = gating_kernel(x, gi_w, pi_w, gi_b, pi_b, ni_w, ni_b)
    # Apply mask
    x = x * mask.unsqueeze(-1)

    # Split input and cast to float
    with torch.autocast("cuda", enabled=False):
        a1, b1, a2, b2 = torch.chunk(x.float(), 4, dim=-1)

        # Triangular projection
        x1 = torch.einsum("bikd,bjkd->bijd", a1, b1)
        x2 = torch.einsum("bkid,bkjd->bijd", a2, b2)

        # Merge outputs
        x = torch.cat([x1, x2], dim=-1).to(x.dtype)

    # Output gating: D / 2 -> D
    x = gating_kernel(x, go_w, po_w, go_b, po_b, no_w, no_b)
    return x


class TransitionUpdate(nn.Module):
    """Perform a two-layer MLP with a residual connection."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        kernels: bool = False,
    ):
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the projection, default 512
        kernel: bool
            Whether to use the kernel function, default False

        """
        super().__init__()

        # Set forward function
        self.fn = mlp_kernel_func if kernels else mlp

        # Initialize parameters
        self.wn = nn.Parameter(torch.empty(dim))
        self.bn = nn.Parameter(torch.empty(dim))
        self.w1 = nn.Parameter(torch.empty(hidden, dim))
        self.b1 = nn.Parameter(torch.empty(hidden))
        self.w2 = nn.Parameter(torch.empty(dim, hidden))
        self.b2 = nn.Parameter(torch.empty(dim))

        init.bias_init_one_(self.wn)
        init.bias_init_zero_(self.bn)

        init.he_normal_init_(self.w1)
        init.final_init_(self.w2)

        init.bias_init_zero_(self.b1)
        init.bias_init_zero_(self.b2)

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)

        Returns
        ----------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        return self.fn(
            x,
            self.w1,
            self.w2,
            self.b1,
            self.b2,
            self.wn,
            self.bn,
        )


class TriangularUpdate(nn.Module):
    """Perform a triangular update.

    This module differs from the original multiplicative
    update introduced in AlphaFold2 in several ways. First,
    we merge the incoming and outgoing layers in a single
    update. Second, and related to the  above change, we
    down-project the input gate to D // 4, thus reducing the
    cost of the inner matmul. Third, we modify the output
    gate to be a function of the output instead of the
    normalized intput, which allows us to use the same
    gating kernel for both the input and output gates.

    """

    def __init__(self, dim: int = 128, kernels: bool = False):
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        kernels: bool
            Whether to use the kernel function, default False

        """
        super().__init__()

        # Set forward function
        self.fn = triangular_kernel_func if kernels else triangular

        # Initialize parameters
        self.ni_w = nn.Parameter(torch.empty(dim))
        self.ni_b = nn.Parameter(torch.empty(dim))
        self.pi_w = nn.Parameter(torch.empty(dim, dim))
        self.gi_w = nn.Parameter(torch.empty(dim, dim))
        self.pi_b = nn.Parameter(torch.empty(dim))
        self.gi_b = nn.Parameter(torch.empty(dim))

        self.no_w = nn.Parameter(torch.empty(dim // 2))
        self.no_b = nn.Parameter(torch.empty(dim // 2))
        self.po_w = nn.Parameter(torch.empty(dim, dim // 2))
        self.go_w = nn.Parameter(torch.empty(dim, dim // 2))
        self.po_b = nn.Parameter(torch.empty(dim))
        self.go_b = nn.Parameter(torch.empty(dim))

        init.bias_init_one_(self.ni_w)
        init.bias_init_zero_(self.ni_b)

        init.lecun_normal_init_(self.pi_w)
        init.gating_init_(self.gi_w)

        init.bias_init_zero_(self.pi_b)
        init.bias_init_one_(self.gi_b)

        init.bias_init_one_(self.no_w)
        init.bias_init_zero_(self.no_b)

        init.final_init_(self.po_w)
        init.gating_init_(self.go_w)

        init.bias_init_zero_(self.po_b)
        init.bias_init_one_(self.go_b)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        ----------
        torch.Tensor
            The output data of shape (B, N, N, D)

        """
        return self.fn(
            x,
            mask,
            self.pi_w,
            self.gi_w,
            self.pi_b,
            self.gi_b,
            self.po_w,
            self.go_w,
            self.po_b,
            self.go_b,
            self.ni_w,
            self.ni_b,
            self.no_w,
            self.no_b,
        )


class Block(nn.Module):
    """Perform a MiniFormer block."""

    def __init__(self, dim: int = 128, kernels: bool = False):
        """Initialize a MiniFormer block.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        kernels:
            Whether to use the kernel function, default False

        """
        super().__init__()
        self.triangular = TriangularUpdate(dim, kernels)
        self.transition = TransitionUpdate(dim, dim * 4, kernels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x1: torch.Tensor
            The input data of shape (B, N, N, D)
        x2: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        ----------
        torch.Tensor
            The output data of shape (B, N, N, D)

        """
        x = x + self.triangular(x, mask)
        x = x + self.transition(x)
        return x


class MiniFormer(nn.Module):
    """The MiniFormer module."""

    def __init__(
        self,
        dim: int = 128,
        blocks: int = 48,
        kernels: bool = False,
    ):
        """Initialize a MiniFormer model.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        blocks: int
            The number of reversible blocks, default 48
        kernels: bool
            Whether to use the kernel function, default False

        """
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, kernels) for _ in range(blocks)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass/

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        torch.Tensor
            The output data of shape (B, N, N, D)

        """
        for block in self.blocks:
            x = block(x, mask)
        return x
