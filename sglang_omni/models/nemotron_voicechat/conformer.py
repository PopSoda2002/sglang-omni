import math
import torch
from torch import nn
from einops import rearrange

SUBSAMPLING_KERNEL_SIZE = 3
SUBSAMPLING_STRIDE = 2

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, *, groups=1):
        super().__init__(in_channels, out_channels, kernel_size=SUBSAMPLING_KERNEL_SIZE, stride=SUBSAMPLING_STRIDE, padding=0, groups=groups)
        self.left_padding = SUBSAMPLING_KERNEL_SIZE - 1
        self.right_padding = SUBSAMPLING_STRIDE - 1

    def forward(self, input_BCTM):
        padded_BCTM = nn.functional.pad(input_BCTM, (self.left_padding, self.right_padding, self.left_padding, self.right_padding))
        convolved_BCTM = super().forward(padded_BCTM)
        return convolved_BCTM

class ConvSubsampling(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_mels = int(config["feat_in"])
        hidden_size = int(config["d_model"])
        channels = int(config["subsampling_conv_channels"])
        factor = int(config["subsampling_factor"])
        num_stages = int(math.log2(factor))

        subsampling_layers: list[nn.Module] = [
            CausalConv2d(in_channels=1, out_channels=channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_stages - 1):
            subsampling_layers.extend((
                CausalConv2d(in_channels=channels, out_channels=channels, groups=channels),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1,),
                nn.ReLU(inplace=True),
            ))

        self.conv = nn.Sequential(*subsampling_layers)
        left_padding = SUBSAMPLING_KERNEL_SIZE - 1
        right_padding = SUBSAMPLING_STRIDE - 1
        for _ in range(num_stages):
            num_mels = (num_mels + left_padding + right_padding - SUBSAMPLING_KERNEL_SIZE) // SUBSAMPLING_STRIDE + 1
        self.out = nn.Linear(in_features=channels * num_mels, out_features=hidden_size)

    def forward(self, features_BTM):
        hidden_BCTM = rearrange(features_BTM, "b t m -> b 1 t m")
        hidden_BCTM = self.conv(hidden_BCTM)
        hidden_BTM = rearrange(hidden_BCTM, "b c t m -> b t (c m)")
        hidden_BTM = self.out(hidden_BTM)
        return hidden_BTM

POINTWISE_CONV_KERNEL_SIZE = 1

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, *, kernel_size, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, groups=groups, bias=bias)
        self.left_padding = kernel_size - 1

    def forward(self, input_BDT):
        padded_BDT = nn.functional.pad(input_BDT, (self.left_padding, 0), mode="constant", value=0.0)
        convolved_BDT = super().forward(padded_BDT)
        return convolved_BDT

class ConformerConvolution(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        kernel_size = int(config["conv_kernel_size"])
        use_bias = bool(config["use_bias"])
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=2 * hidden_size,
            kernel_size=POINTWISE_CONV_KERNEL_SIZE,
            bias=use_bias,
        )
        self.depthwise_conv = CausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=use_bias,
        )
        # Name is misleading, it's not batch norm, but layer norm
        self.batch_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=POINTWISE_CONV_KERNEL_SIZE,
            bias=use_bias,
        )

    def forward(self, hidden_BTD):
        hidden_BDT = rearrange(hidden_BTD, "b t d -> b d t")
        gates_BDT = self.pointwise_conv1(hidden_BDT)
        hidden_BDT = nn.functional.glu(gates_BDT, dim=1)
        hidden_BDT = self.depthwise_conv(hidden_BDT)
        hidden_BTD = rearrange(hidden_BDT, "b d t -> b t d")
        hidden_BTD = self.batch_norm(hidden_BTD)
        hidden_BDT = rearrange(hidden_BTD, "b t d -> b d t")
        hidden_BDT = self.activation(hidden_BDT)
        hidden_BDT = self.pointwise_conv2(hidden_BDT)
        output_BTD = rearrange(hidden_BDT, "b d t -> b t d")
        return output_BTD

class ConformerFeedForward(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        expansion = int(config["ff_expansion_factor"])
        intermediate_size = int(hidden_size * expansion)
        use_bias = bool(config["use_bias"])
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=intermediate_size, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(in_features=intermediate_size, out_features=hidden_size, bias=use_bias)

    def forward(self, hidden_BTD):
        expanded_BTE = self.linear1(hidden_BTD)
        expanded_BTE = self.activation(expanded_BTE)
        hidden_BTD = self.linear2(expanded_BTE)
        return hidden_BTD