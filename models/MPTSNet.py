import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mptsnet_layer import (
    DataEmbedding,
    clsWindowTransformer,
    Inception_CBAM,
    clsTransformer,
    DataEmbedding_v1,
    Transformer,
    WindowTransformer,
)
from utils.mptsnet_utils import fft_find_each_amplitude


PERIOD_FILE = (
    "/home/remote/s225150779/GitHub/Multi-variate-TSC-NIPS/"
    "uea_top5_periods_sgn_wo_duplicates.txt"
)


def load_candidate_period_map(file_path):
    """
    Read pre-computed Top-5 period candidates.

    Expected format:
        Dataset: CharacterTrajectories
        --> Top-5 Candidates: [p1, p2, p3, p4, p5]
        --> Final Period: ...

    MPTSNet uses all five candidates.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"[MPTSNet] Period file not found: {file_path}"
        )

    candidate_map = {}
    current_dataset = None

    with open(file_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            dataset_match = re.match(r"Dataset:\s*(.+)", line)
            if dataset_match:
                current_dataset = dataset_match.group(1).strip()
                continue

            if current_dataset is None:
                continue

            candidate_match = re.search(
                r"Top-5 Candidates:\s*\[(.*?)\]",
                line,
            )

            if candidate_match:
                candidate_text = candidate_match.group(1)
                candidates = [
                    int(x.strip())
                    for x in candidate_text.split(",")
                    if x.strip()
                ]
                candidate_map[current_dataset] = candidates

    return candidate_map


CANDIDATE_PERIOD_MAP = load_candidate_period_map(PERIOD_FILE)


def get_mptsnet_periods(dataset_name):
    """Return exactly five pre-computed candidate periods."""
    if dataset_name not in CANDIDATE_PERIOD_MAP:
        available_preview = list(CANDIDATE_PERIOD_MAP.keys())[:10]
        raise KeyError(
            f"[MPTSNet] Dataset '{dataset_name}' was not found in "
            f"{PERIOD_FILE}. First available keys: {available_preview}"
        )

    periods = CANDIDATE_PERIOD_MAP[dataset_name]

    if len(periods) < 5:
        raise ValueError(
            f"[MPTSNet] Expected at least 5 period candidates for "
            f"'{dataset_name}', but found {periods}"
        )

    periods = [int(p) for p in periods[:5]]

    if any(p <= 0 for p in periods):
        raise ValueError(
            f"[MPTSNet] All periods must be positive integers. "
            f"Got {periods} for '{dataset_name}'."
        )

    return periods


class PeriodicBlock(nn.Module):
    def __init__(
        self,
        flag,
        periods,
        seq_length,
        embed_dim,
        embed_dim_t,
        num_heads,
        ff_dim,
        num_layers,
    ):
        super(PeriodicBlock, self).__init__()

        self.periods = periods
        self.embed_dim = embed_dim

        self.cnn = nn.Sequential(
            Inception_CBAM(embed_dim, 1024),
            nn.GELU(),
            Inception_CBAM(1024, embed_dim),
        )

        if flag:
            self.transformer = Transformer(
                embed_dim,
                seq_length,
                embed_dim,
                num_heads,
                ff_dim,
                num_layers,
            )

            self.transformers = nn.ModuleList(
                [
                    WindowTransformer(
                        embed_dim * period,
                        (seq_length // period) + 1,
                        embed_dim_t,
                        num_heads,
                        ff_dim,
                        num_layers,
                    )
                    for period in periods
                ]
            )
        else:
            self.transformer = clsTransformer(
                seq_length,
                embed_dim,
                num_heads,
                ff_dim,
                num_layers,
            )

            self.transformers = nn.ModuleList(
                [
                    clsWindowTransformer(
                        embed_dim * period,
                        (seq_length // period) + 1,
                        embed_dim_t,
                        num_heads,
                        ff_dim,
                        num_layers,
                    )
                    for period in periods
                ]
            )

    def forward(self, x):
        B = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]

        time_point_features = self.transformer(x)

        global_features = []
        amplitudes = []

        for i, period in enumerate(self.periods):
            x_fft = x.permute(0, 2, 1).detach().cpu().numpy()
            amplitudes.append(
                fft_find_each_amplitude(x_fft, period)
            )

            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros(
                    [B, C, length - T],
                    device=x.device,
                    dtype=x.dtype,
                )
                out = torch.cat([x, padding], dim=2)
            else:
                length = T
                out = x

            num_period = length // period

            out = out.reshape(
                B,
                C,
                period,
                num_period,
            ).contiguous()

            local_features = []

            for j in range(num_period):
                feature = self.cnn(out[:, :, :, j])
                local_features.append(feature)

            local_features = torch.stack(
                local_features,
                dim=-1,
            )

            local_features = out + local_features

            local_features = local_features.reshape(
                B,
                -1,
                num_period,
            )

            global_feature = self.transformers[i](
                local_features
            )

            global_feature = global_feature.reshape(
                B,
                self.embed_dim,
                -1,
            ).contiguous()

            global_feature = global_feature[:, :, :T]
            global_features.append(global_feature)

        amplitudes = torch.cat(amplitudes, dim=1)

        global_features = torch.stack(
            global_features,
            dim=-1,
        )

        weights = torch.softmax(
            amplitudes,
            dim=1,
        )

        period_weight = (
            weights.unsqueeze(1)
            .unsqueeze(1)
            .repeat(
                1,
                self.embed_dim,
                T,
                1,
            )
            .to(x.device)
        )

        res = torch.sum(
            global_features * period_weight,
            dim=-1,
        )

        res = res + time_point_features + x

        return res


class Model(nn.Module):
    """
    TSLib-compatible MPTSNet.

    TSLib classification inputs are [B, T, C].
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        dataset_name = str(configs.model_id)
        periods = get_mptsnet_periods(dataset_name)

        print("=" * 70)
        print(f"[MPTSNet] Dataset           : {dataset_name}")
        print(f"[MPTSNet] Period file       : {PERIOD_FILE}")
        print(f"[MPTSNet] Top-5 periods     : {periods}")
        print("=" * 70)

        num_channels = int(configs.enc_in)
        seq_length = int(configs.seq_len)
        num_classes = int(configs.num_class)

        embed_dim = max(
            min(num_channels * 4, 256),
            64,
        )

        embed_dim_t = max(
            min(embed_dim * 4, 512),
            256,
        )

        num_heads = 4
        ff_dim = 256
        num_layers = 1

        flag = dataset_name == "PEMS-SF"

        print(f"[MPTSNet] seq_length        : {seq_length}")
        print(f"[MPTSNet] num_channels      : {num_channels}")
        print(f"[MPTSNet] num_classes       : {num_classes}")
        print(f"[MPTSNet] embed_dim         : {embed_dim}")
        print(f"[MPTSNet] embed_dim_t       : {embed_dim_t}")
        print(f"[MPTSNet] DataEmbedding_v1  : {flag}")
        print("=" * 70)

        if flag:
            self.enc_embedding = DataEmbedding_v1(
                num_channels,
                embed_dim,
                dropout=0.1,
            )
        else:
            self.enc_embedding = DataEmbedding(
                num_channels,
                embed_dim,
                seq_length,
                dropout=0.1,
            )

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.model = nn.ModuleList(
            [
                PeriodicBlock(
                    flag,
                    periods,
                    seq_length,
                    embed_dim,
                    embed_dim_t,
                    num_heads,
                    ff_dim,
                    num_layers,
                )
                for _ in range(2)
            ]
        )

        self.activation = F.gelu
        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(
            seq_length * embed_dim,
            num_classes,
        )

    def forward(
        self,
        x,
        padding_mask=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        # TSLib input: [B, T, C]
        x = self.enc_embedding(x)   # [B, T, E]
        x = x.permute(0, 2, 1)     # [B, E, T]

        for i in range(2):
            x = self.layer_norm(
                self.model[i](x)
                .permute(0, 2, 1)
            ).permute(0, 2, 1)

        x = self.activation(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        output = self.fc(x.float())

        return output
