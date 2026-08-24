from __future__ import annotations

import torch
from torch import nn

class AddFusion(nn.Module):
    def __init__(self, *, text_weight, user_weight, function_weight):
        super().__init__()
        self.text_weight = float(text_weight)
        self.user_weight = float(user_weight)
        self.function_weight = float(function_weight)

    def forward(self, acoustic, text, function: torch.Tensor | None = None) -> torch.Tensor:
        output = (self.user_weight * acoustic) + (self.text_weight * text)
        if function is not None:
            output = output + (self.function_weight * function)
        return output