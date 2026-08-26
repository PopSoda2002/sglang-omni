import math
import torch
from torch import nn
from einops import rearrange, einsum

from sglang_omni.models.nemotron_voicechat.preprocess import LogMelFeatures

SUBSAMPLING_KERNEL_SIZE = 3
SUBSAMPLING_STRIDE = 2

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, *, groups=1):
        super().__init__(in_channels, out_channels, kernel_size=SUBSAMPLING_KERNEL_SIZE, stride=SUBSAMPLING_STRIDE, padding=0, groups=groups)
        self.time_padding = (SUBSAMPLING_KERNEL_SIZE - 1, 0)
        self.freq_padding = (SUBSAMPLING_KERNEL_SIZE - 1, SUBSAMPLING_STRIDE - 1)

    def forward(self, input_BCTM):
        padded_BCTM = nn.functional.pad(input_BCTM, (*self.freq_padding, *self.time_padding))
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
        flattened_BTF = rearrange(hidden_BCTM, "b c t m -> b t (c m)")
        subsampled_BTD = self.out(flattened_BTF)
        return subsampled_BTD

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

class RelPositionalEncoding(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        inverse_freq_H = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * -(math.log(10000.0) / hidden_size)
        )
        self.register_buffer("inverse_freq_H", inverse_freq_H, persistent=False)

    def forward(self, hidden_BTD):
        length = hidden_BTD.shape[1]
        positions_P = torch.arange(length - 1, -length, -1.0, device=hidden_BTD.device)
        angles_PH = einsum(positions_P, self.inverse_freq_H, "p, h -> p h")
        encoding_PD = rearrange(torch.stack((angles_PH.sin(), angles_PH.cos()), dim=-1), "p h two -> p (h two)")
        return rearrange(encoding_PD, "p d -> 1 p d").to(hidden_BTD.dtype)

class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        self.num_heads = int(config["n_heads"])
        self.head_size = hidden_size // self.num_heads
        use_bias = bool(config["use_bias"])
        self.left_context, self.right_context = (int(value) for value in config["att_context_size"])

        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_out = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    def _relative_shift(self, scores_BHTP):
        batch, heads, length, positions = scores_BHTP.shape
        padded_BHTP = nn.functional.pad(scores_BHTP, (1, 0))
        walked_BHPT = padded_BHTP.view(batch, heads, -1, length)
        shifted_BHTP = walked_BHPT[:, :, 1:].view(batch, heads, length, positions)
        return shifted_BHTP[:, :, :, :length]


    def _context_mask(self, length, device):
        positions_T = torch.arange(length, device=device)
        offsets_TT = positions_T[None, :] - positions_T[:, None]
        allowed_TT = (offsets_TT <= self.right_context) & (offsets_TT >= -self.left_context)
        return rearrange(allowed_TT, "q k -> 1 1 q k")

    def forward(self, hidden_BTD, encoding_BPD):
        queries_BHTS = rearrange(self.linear_q(hidden_BTD), "b t (h s) -> b h t s", h=self.num_heads)
        keys_BHTS = rearrange(self.linear_k(hidden_BTD), "b t (h s) -> b h t s", h=self.num_heads)
        values_BHTS = rearrange(self.linear_v(hidden_BTD), "b t (h s) -> b h t s", h=self.num_heads)
        positions_BHPS = rearrange(self.linear_pos(encoding_BPD), "b p (h s) -> b h p s", h=self.num_heads)

        content_BHTS = queries_BHTS + rearrange(self.pos_bias_u, "h s -> 1 h 1 s")
        position_BHTS = queries_BHTS + rearrange(self.pos_bias_v, "h s -> 1 h 1 s")
        content_scores_BHTT = einsum(content_BHTS, keys_BHTS, "b h q s, b h k s -> b h q k")
        position_scores_BHTP = einsum(position_BHTS, positions_BHPS, "b h q s, b h p s -> b h q p")
        scores_BHTT = (content_scores_BHTT + self._relative_shift(position_scores_BHTP)) / math.sqrt(self.head_size)

        allowed_TT = self._context_mask(hidden_BTD.shape[1], hidden_BTD.device)
        scores_BHTT = scores_BHTT.masked_fill(~allowed_TT, torch.finfo(scores_BHTT.dtype).min)
        weights_BHTT = scores_BHTT.softmax(dim=-1)
        attended_BHTS = einsum(weights_BHTT, values_BHTS, "b h q k, b h k s -> b h q s")
        return self.linear_out(rearrange(attended_BHTS, "b h t s -> b t (h s)"))

FEED_FORWARD_RESIDUAL_SCALE = 0.5

class ConformerLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["d_model"])
        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.feed_forward1 = ConformerFeedForward(config)
        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = RelPositionMultiHeadAttention(config)
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.conv = ConformerConvolution(config)
        self.norm_feed_forward2 = nn.LayerNorm(hidden_size)
        self.feed_forward2 = ConformerFeedForward(config)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, hidden_BTD, encoding_BPD):
        hidden_BTD = hidden_BTD + FEED_FORWARD_RESIDUAL_SCALE * self.feed_forward1(self.norm_feed_forward1(hidden_BTD))
        hidden_BTD = hidden_BTD + self.self_attn(self.norm_self_att(hidden_BTD), encoding_BPD)
        hidden_BTD = hidden_BTD + self.conv(self.norm_conv(hidden_BTD))
        hidden_BTD = hidden_BTD + FEED_FORWARD_RESIDUAL_SCALE * self.feed_forward2(self.norm_feed_forward2(hidden_BTD))
        return self.norm_out(hidden_BTD)

class ConformerEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.pre_encode = ConvSubsampling(config)
        self.pos_enc = RelPositionalEncoding(config)
        self.layers = nn.ModuleList(ConformerLayer(config) for _ in range(int(config["n_layers"])))
        self.xscaling = bool(config["xscaling"])

    def forward(self, features_BTM):
        hidden_BTD = self.pre_encode(features_BTM)
        if self.xscaling:
            hidden_BTD = hidden_BTD * math.sqrt(hidden_BTD.shape[-1])
        encoding_BPD = self.pos_enc(hidden_BTD)
        for layer in self.layers:
            hidden_BTD = layer(hidden_BTD, encoding_BPD)
        return hidden_BTD

class AudioPerception(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.preprocessor = LogMelFeatures(config["preprocessor"])
        self.encoder = ConformerEncoder(config["encoder"])
        self.proj = nn.Linear(int(config["encoder"]["d_model"]), int(config["output_dim"]))

    def forward(self, waveform_BL):
        features_BTM = self.preprocessor(waveform_BL)
        hidden_BTD = self.encoder(features_BTM)
        return self.proj(hidden_BTD)
