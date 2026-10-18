from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - depends on local CUDA build.
    Mamba = None


def require_mamba() -> None:
    if Mamba is None:
        raise ImportError(
            "mamba_ssm is required for Mamba models. Install causal-conv1d and mamba-ssm "
            "in the active PyTorch/CUDA environment."
        )


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, activation: str = "relu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            act,
        )

    def forward(self, x):
        return self.block(x)


class SafeAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    """Adaptive average pooling with a CUDA-safe divisible-length path."""

    def forward(self, x):
        output_len = self.output_size
        if isinstance(output_len, (tuple, list)):
            output_len = output_len[0]
        if x.shape[-1] % output_len == 0:
            group_size = x.shape[-1] // output_len
            return x.reshape(*x.shape[:-1], output_len, group_size).mean(dim=-1)
        return super().forward(x)


class DeepFeatureResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class HybridCNNMambaEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, num_cnn_blocks: int = 3, num_mamba_layers: int = 3, is_sequence: bool = True):
        super().__init__()
        require_mamba()
        self.is_sequence = is_sequence
        if self.is_sequence:
            self.downsampler = nn.Sequential(
                DownsampleBlock(in_channels, 48, kernel_size=5, stride=5, activation="gelu"),
                DownsampleBlock(48, 96, kernel_size=5, stride=5, activation="gelu"),
                DownsampleBlock(96, d_model, kernel_size=5, stride=5, activation="gelu"),
                DownsampleBlock(d_model, d_model, kernel_size=5, stride=4, activation="gelu"),
                DownsampleBlock(d_model, d_model, kernel_size=5, stride=4, activation="gelu"),
            )
        else:
            self.downsampler = nn.Sequential(nn.Conv1d(in_channels, d_model, kernel_size=1), nn.GELU())
        self.cnn_feature_extractor = nn.Sequential(*[DeepFeatureResBlock(d_model) for _ in range(num_cnn_blocks)])
        self.mamba_layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(num_mamba_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.downsampler(x)
        x = self.cnn_feature_extractor(x).transpose(1, 2)
        x = self.norm(x)
        for mamba_layer in self.mamba_layers:
            x_fw = mamba_layer(x)
            x_bw = torch.flip(mamba_layer(torch.flip(x, dims=[1])), dims=[1])
            x = x + x_fw + x_bw
        return x.transpose(1, 2)


class SimpleFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, seq_emb, omics_emb):
        return self.fusion_conv(torch.cat([seq_emb, omics_emb], dim=1))


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class PracticalOuterProductDecoder(nn.Module):
    def __init__(self, in_channels: int = 128, bottleneck_channels: int = 16, num_blocks: int = 4):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1)
        self.initial_conv = nn.Conv2d(bottleneck_channels * 2, 64, 1)
        self.res_blocks = nn.Sequential(*[ConvResBlock(64) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, fused_embedding):
        _, _, length = fused_embedding.shape
        bottleneck_emb = self.bottleneck(fused_embedding)
        x1 = bottleneck_emb.unsqueeze(3).expand(-1, -1, length, length)
        x2 = bottleneck_emb.unsqueeze(2).expand(-1, -1, length, length)
        x = self.initial_conv(torch.cat([x1, x2], dim=1))
        x = self.final_conv(self.res_blocks(x))
        return (x + x.transpose(-2, -1)) / 2


class CNNMambaHybridModelLegacy(nn.Module):
    """Original GM12878 Figure 04 model matching output_GM12878_train_64stride/best_model.pth."""

    def __init__(
        self,
        seq_in_channels: int = 4,
        num_omics_features: int = 4,
        d_model: int = 128,
        fusion_out_dim: int = 128,
        num_cnn_blocks: int = 3,
        num_mamba_layers: int = 3,
        decoder_bottleneck_channels: int = 48,
        decoder_num_blocks: int = 6,
    ):
        super().__init__()
        self.seq_encoder = HybridCNNMambaEncoder(seq_in_channels, d_model, num_cnn_blocks, num_mamba_layers, is_sequence=True)
        self.omics_encoder = HybridCNNMambaEncoder(num_omics_features, d_model, num_cnn_blocks, num_mamba_layers, is_sequence=False)
        self.fusion_module = SimpleFusion(in_channels=d_model, out_channels=fusion_out_dim)
        self.decoder = PracticalOuterProductDecoder(fusion_out_dim, decoder_bottleneck_channels, decoder_num_blocks)

    def encode_fused(self, seq, omics):
        seq_emb = self.seq_encoder(seq)
        omics_emb = self.omics_encoder(omics)
        return self.fusion_module(seq_emb, omics_emb)

    def forward(self, seq, omics):
        return self.decoder(self.encode_fused(seq, omics)).squeeze(1)


def build_model(
    model_type: str,
    num_omics_features: int,
    target_len: int = 512,
    *,
    ablate_dna: bool = False,
    ablate_omics_all: bool = False,
    decoder_variant: str = "diagonal",
    encoder_variant: str = "single",
):
    legacy_aliases = {"gm12878_legacy", "mamba_legacy", "figure04_legacy", "legacy", "main"}
    if model_type in legacy_aliases:
        return CNNMambaHybridModelLegacy(num_omics_features=num_omics_features)
    raise ValueError(
        "This project keeps only the legacy CNNMambaHybridModelLegacy implementation; "
        f"unsupported model_type={model_type!r}."
    )
