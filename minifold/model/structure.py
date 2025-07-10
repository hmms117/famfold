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

from typing import Tuple
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from minifold.utils import init
from minifold.utils.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from minifold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from minifold.utils.rigid_utils import Rigid
from minifold.utils.tensor_utils import dict_multimap, permute_final_dims


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_width: int):
        """Initializes the Attention module.

        Parameters
        ----------
        dim: int
            Input dimension
        num_heads: int
            Number of attention heads
        head_width: int
            Width of each attention head

        """
        super().__init__()
        assert dim == num_heads * head_width

        self.dim = dim
        self.num_heads = num_heads
        self.head_width = head_width
        self.rescale_factor = self.head_width**-0.5

        self.layer_norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.g_proj = nn.Linear(dim, dim)

        torch.nn.init.zeros_(self.o_proj.bias)
        torch.nn.init.zeros_(self.g_proj.weight)
        torch.nn.init.ones_(self.g_proj.bias)

    def forward(self, x: Tensor, bias: Tensor, mask: Tensor) -> Tensor:
        """Forward pass of the Attention module.

        Parameters
        ----------
        x: Tensor
            Input tensor of shape (B, N, D)
        bias: Tensor
            External attention bias tensor of shape (B, H, N, N)
        mask: Tensor
            Mask tensor of shape (B, N)

        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D)

        """
        # Layer norm
        x = self.layer_norm(x)

        t = rearrange(self.proj(x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k, v = t.chunk(3, dim=-1)

        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias.
        a = a + bias

        # Do not attend to padding tokens.
        mask = repeat(mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2])
        a = a.masked_fill(mask == 0, -np.inf)
        a = F.softmax(a, dim=-1)

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)
        y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initializes the MLP module.

        Parameters
        ----------
        in_dim:
            Input dimension
        out_dim:
            Output dimension

        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

        init.he_normal_init_(self.mlp[1].weight)
        init.final_init_(self.mlp[3].weight)
        init.bias_init_zero_(self.mlp[1].bias)
        init.bias_init_zero_(self.mlp[3].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP module.

        Parameters
        ----------
        x:
            Input tensor of shape (..., D_in)

        Returns
        -------
        Tensor:
            Output tensor of shape (..., D_out)

        """
        return self.mlp(x)


class AngleResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        init.he_normal_init_(self.mlp[1].weight)
        init.final_init_(self.mlp[3].weight)
        init.bias_init_zero_(self.mlp[1].bias)
        init.bias_init_zero_(self.mlp[3].bias)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return a + self.mlp(a)


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = nn.Linear(self.c_in, self.c_hidden)
        self.linear_initial = nn.Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(dim=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = nn.Linear(self.c_hidden, self.no_angles * 2)

        init.lecun_normal_init_(self.linear_in.weight)
        init.lecun_normal_init_(self.linear_initial.weight)
        init.final_init_(self.linear_out.weight)

        init.bias_init_zero_(self.linear_in.bias)
        init.bias_init_zero_(self.linear_initial.bias)
        init.bias_init_zero_(self.linear_out.bias)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_resnet: int,
        head_dim: int,
        no_heads: int,
        no_blocks: int,
        no_resnet_blocks: int,
        no_angles: int,
        trans_scale_factor: float,
        epsilon: float,
        inf: float,
    ):
        """Initializes the StructureModule.

        Parameters
        ----------
        c_s:
            Single representation channel dimension
        c_z:
            Pair representation channel dimension
        c_resnet:
            Angle resnet hidden channel dimension
        head_dim:
            Dimension of each transformer head
        no_heads:
            Number of transformer heads
        no_blocks:
            Number of structure module blocks
        no_resnet_blocks:
            Number of blocks in the angle resnet
        no_angles:
            Number of angles to generate in the angle resnet
        trans_scale_factor:
            Scale of output coordinates
        epsilon:
            Small number used in angle resnet normalization
        inf:
            Large number used for attention masking

        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_resnet = c_resnet
        self.no_heads = no_heads
        self.head_dim = head_dim
        self.no_blocks = no_blocks
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        self.layer_norm_s = nn.LayerNorm(self.c_s)
        self.layer_norm_z = nn.LayerNorm(self.c_z)
        self.linear_in = nn.Linear(self.c_s, self.c_s)
        self.linear_b = nn.Linear(self.c_z, self.no_blocks * self.no_heads)

        self.attn = nn.ModuleList(
            [
                Attention(self.c_s, self.no_heads, self.head_dim)
                for _ in range(self.no_blocks)
            ]
        )
        self.transitions = nn.ModuleList(
            [MLP(self.c_s, self.c_s) for _ in range(self.no_blocks)]
        )

        self.bb_update = nn.Linear(self.c_s, 9)
        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

        # Initialize the weights
        init.lecun_normal_init_(self.linear_in.weight)
        init.bias_init_zero_(self.linear_in.bias)
        init.lecun_normal_init_(self.bb_update.weight)
        init.bias_init_zero_(self.bb_update.bias)
        init.lecun_normal_init_(self.linear_b.weight)
        init.bias_init_zero_(self.linear_b.bias)

        # Initialize the buffers
        frames = torch.tensor(restype_rigid_group_default_frame)
        groups = torch.tensor(restype_atom14_to_rigid_group)
        atom_mask = torch.tensor(restype_atom14_mask)
        positions = torch.tensor(restype_atom14_rigid_group_positions)

        self.register_buffer("default_frames", frames, persistent=False)
        self.register_buffer("group_idx", groups, persistent=False)
        self.register_buffer("atom_mask", atom_mask, persistent=False)
        self.register_buffer("lit_positions", positions, persistent=False)

    def forward(
        self,
        s,
        z,
        aatype,
        mask,
    ):
        # Input projection
        s = self.layer_norm_s(s)
        s_initial = s
        s = self.linear_in(s)

        # Pairwise bias
        B, N = s.shape[:2]
        z = self.layer_norm_z(z)
        b = self.linear_b(z)
        b = permute_final_dims(b, (2, 0, 1))
        b = b.reshape(B, self.no_blocks, self.no_heads, N, N)

        # Apply transformer layers
        outputs = []
        for i in range(self.no_blocks):
            s = s + self.attn[i](s, b[:, i], mask)
            s = s + self.transitions[i](s)

        # Predict angles
        unnormalized_angles, angles = self.angle_resnet(s, s_initial)

        # Predict positions
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        with torch.autocast(device.type, enabled=False):
            # Predict frames
            n, ca, c = self.bb_update(s.float()).chunk(3, dim=-1)
            rigids = Rigid.make_transform_from_reference(n, ca, c, eps=1e-7)
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            # Compute all positions
            all_frames_to_global = torsion_angles_to_frames(
                scaled_rigids, angles, aatype, self.default_frames
            )
            pred_xyz = frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
                self.default_frames,
                self.group_idx,
                self.atom_mask,
                self.lit_positions,
            )
            outputs.append(
                {
                    "angles": angles,
                    "unnormalized_angles": unnormalized_angles,
                    "frames": scaled_rigids.to_tensor_4x4(),
                    "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                    "positions": pred_xyz,
                    "states": s,
                }
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        return outputs
