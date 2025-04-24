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

import torch.nn as nn

from minifold.utils import init
from minifold.train.loss import compute_plddt


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden):
        super().__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear_1 = nn.Linear(self.c_in, self.c_hidden)
        self.linear_2 = nn.Linear(self.c_hidden, self.c_hidden)
        self.linear_3 = nn.Linear(self.c_hidden, self.no_bins)

        init.he_normal_init_(self.linear_1.weight)
        init.he_normal_init_(self.linear_2.weight)
        init.final_init_(self.linear_3.weight)

        init.bias_init_zero_(self.linear_1.bias)
        init.bias_init_zero_(self.linear_2.bias)
        init.bias_init_zero_(self.linear_3.bias)

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        return s


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )
        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits
        aux_out["plddt"] = compute_plddt(lddt_logits)
        return aux_out
