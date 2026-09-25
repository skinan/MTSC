import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re


def load_period_map(file_path):
    period_map = {}
    candidate_map = {}
    current_dataset = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            dataset_match = re.match(r"Dataset:\s*(.+)", line)
            period_match = re.search(r"Final Period:\s*(\d+)", line)
            candidate_match = re.search(r"Top-5 Candidates:\s*\[(.*?)\]", line)

            if dataset_match:
                current_dataset = dataset_match.group(1)

            elif candidate_match and current_dataset:
                # Convert string list → actual list of ints
                candidates = list(map(int, candidate_match.group(1).split(",")))
                candidate_map[current_dataset] = candidates

            elif period_match and current_dataset:
                period_map[current_dataset] = int(period_match.group(1))

    return period_map, candidate_map


def get_period(dataset_name, period_map, candidate_map, period_flag):
    if dataset_name not in period_map:
        raise ValueError(f"[ERROR] Period not found for dataset '{dataset_name}'.")
    if period_flag == "min":
        p = min(candidate_map[dataset_name])
    else:
        p = candidate_map[dataset_name][period_flag]
    return p


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, L, D]
        return self.pe[:, :x.size(1)]


class Inception_Block_1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=7):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, 2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])

    def forward(self, x):
        # x: [N, D, P]
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


class InceptionExpert(nn.Module):
    def __init__(self, d_model, num_kernels=7):
        super().__init__()
        self.net = nn.Sequential(
            Inception_Block_1D(d_model, d_model, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_1D(d_model, d_model, num_kernels=num_kernels),
        )
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: [N, D, P]
        return self.bn(self.net(x)) + x


class HybridRouter(nn.Module):
    """
    Per-feature mean + per-feature FFT magnitudes.

    Input:
        x: [B,C,W,D,P]

    Output:
        logits: [B,C,W,num_experts]
    """

    def __init__(
        self,
        d_model,
        num_experts,
        k_freq=1,
        temperature=0.8,
        exclude_dc=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.k_freq = k_freq
        self.temperature = temperature
        self.exclude_dc = exclude_dc
        hidden_dim = self.d_model // 2

        # Every D feature contributes:
        #   1 mean value
        #   k_freq FFT magnitude values
        router_input_dim = (
            d_model * (1 + k_freq)
        )

        self.router_mlp = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(
                router_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                num_experts,
            ),
        )

    def forward(self, x):
        """
        x: [B,C,W,D,P]
        """

        B, C, W, D, P = x.shape

        feature_mean = x.mean(
            dim=-1
        )  # [B,C,W,D]

        feature_mean = feature_mean.unsqueeze(
            -1
        )  # [B,C,W,D,1]

        fft_magnitude = torch.fft.rfft(
            x.float(),
            dim=-1,
        ).abs()  # [B,C,W,D,F]

        # Remove DC because feature_mean already represents
        # the average level.
        if self.exclude_dc:
            fft_magnitude = fft_magnitude[
                ...,
                1:,
            ]

        available_freqs = fft_magnitude.size(-1)

        if self.k_freq > available_freqs:
            raise ValueError(
                f"k_freq={self.k_freq} exceeds "
                f"available FFT bins={available_freqs}."
            )

        # Top-k magnitudes independently for every D feature.
        topk_magnitudes = torch.topk(
            fft_magnitude,
            k=self.k_freq,
            dim=-1,
        ).values  # [B,C,W,D,K]

        # Normalize using the complete spectrum of each feature.
        spectral_total = fft_magnitude.sum(
            dim=-1,
            keepdim=True,
        )

        topk_magnitudes = (
            topk_magnitudes
            / (
                spectral_total
                + 1e-8
            )
        )  # [B,C,W,D,K]

        router_features = torch.cat(
            [
                feature_mean.float(),
                topk_magnitudes,
            ],
            dim=-1,
        )  # [B,C,W,D,1+K]

        router_features = router_features.flatten(
            start_dim=-2
        )  # [B,C,W,D*(1+K)]

        logits = self.router_mlp(
            router_features
        )  # [B,C,W,num_experts]

        return logits


class RoutingSparsifier(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.threshold_proj = nn.Sequential(
            nn.Linear(num_experts, num_experts),
            nn.ReLU(),
            nn.Linear(num_experts, 1)
        )

    def forward(self, weights):
        """
        weights: [B, C, W, E]
        """
        threshold = torch.sigmoid(self.threshold_proj(weights)).squeeze(-1)  # [B,C,W]
        threshold = threshold.unsqueeze(-1)                                 # [B,C,W,1]

        mask = (weights > threshold).float()                                

        # Ensure at least 1 expert (Top-1 Routing)
        top1_idx = torch.argmax(weights, dim=-1, keepdim=True)
        fallback = torch.zeros_like(weights).scatter_(-1, top1_idx, 1.0)
        mask = torch.maximum(mask, fallback)
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return weights


class InterPatchSequenceLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, 1)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, C, W, D, P]
        """
        B, C, W, D, P = x.shape

        pooled = x.mean(dim=-1).reshape(B * C, W, D)  # [BC, W, D]

        res = pooled
        seq_out = pooled.permute(0, 2, 1)             # [BC, D, W]
        seq_out = self.net(seq_out)                   # [BC, D, W]
        seq_out = seq_out.permute(0, 2, 1)            # [BC, W, D]
        seq_out = self.norm(seq_out + res)            # [BC, W, D]
        seq_out = seq_out.reshape(B, C, W, D, 1)      # broadcast over P
        return x + seq_out


class RouterInducedChannelGraph(nn.Module):
    """
    Patch-wise dynamic channel graph induced by the SOFT MoE router.

    Inputs:
        x:            [B, C, W, D, P]
        router_probs: [B, C, W, E]

    For every sample b and patch w, channels are graph nodes. The router
    probabilities provide channel-to-expert assignment strengths R. Rather
    than materializing a C x C adjacency matrix, message passing is computed
    efficiently through latent expert nodes:

        channel -> expert -> channel

    Let H be the patch-pooled channel features [C,D] and R be [C,E].
    Expert prototypes are

        U_e = sum_c R_ce H_c / (sum_c R_ce + eps)

    and the message received by channel c is

        M_c = sum_e R_ce U_e.

    This is equivalent to applying the normalized low-rank channel graph

        A = R diag(1 / (R^T 1)) R^T,

    so M = A H, but avoids explicitly constructing [C,C].
    """

    def __init__(
        self,
        d_model,
        dropout=0.0,
        init_gate=0.1,
        detach_router=False,
        eps=1e-6,
    ):
        super().__init__()

        self.d_model = d_model
        self.detach_router = detach_router
        self.eps = eps

        # Graph message transform (analogous to the learnable W in a GNN).
        self.message_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.message_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Scalar residual gate constrained to (0,1).  A small initialization
        # lets the graph branch start as a mild correction to the base model.
        init_gate = float(min(max(init_gate, 1e-4), 1.0 - 1e-4))
        gate_logit = math.log(init_gate / (1.0 - init_gate))
        self.gate_logit = nn.Parameter(
            torch.tensor(gate_logit, dtype=torch.float32)
        )

        # Lightweight diagnostics; no CxC matrix is stored during training.
        self.last_gate = None
        self.last_expert_mass = None


    def forward(self, x, router_probs):
        """
        Position-preserving router-induced channel graph.

        Inputs
        ------
        x:
            [B, C, W, D, P]

            Full MoE patch representations.

        router_probs:
            [B, C, W, E]

            Soft router probabilities, before RoutingSparsifier.

        Returns
        -------
        out:
            [B, C, W, D, P]

        The router decides WHICH channels communicate,
        while the complete [D,P] representation determines
        WHAT information is communicated.
        """

        if router_probs is None:
            raise ValueError(
                "router_probs must be provided to "
                "RouterInducedChannelGraph."
            )

        B, C, W, D, P = x.shape

        if C <= 1:
            return x

        r = router_probs
        # [B,C,W,E]

        if self.detach_router:
            r = r.detach()

        # How much total routing mass each expert receives
        # from all physical channels.
        expert_mass = r.sum(
            dim=1
        ).clamp_min(
            self.eps
        )
        # [B,W,E]
        expert_nodes = torch.einsum(
            "bcwe,bcwdp->bwedp",
            r,
            x,
        )

        # Normalize each expert by the amount of
        # channel routing mass it received.
        expert_nodes = (
            expert_nodes
            / expert_mass.unsqueeze(-1).unsqueeze(-1)
        )

        message = torch.einsum(
            "bcwe,bwedp->bcwdp",
            r,
            expert_nodes,
        )

        # =====================================================
        # 4. Learnable feature transformation
        # =====================================================

        # message_proj is nn.Linear(D,D), therefore D
        # needs to be the last dimension.
        message = message.permute(
            0, 1, 2, 4, 3
        )
        # [B,C,W,P,D]

        message = self.message_proj(
            message
        )
        # [B,C,W,P,D]

        message = self.message_norm(
            message
        )

        message = self.dropout(
            message
        )

        message = message.permute(
            0, 1, 2, 4, 3
        ).contiguous()

        gate = torch.sigmoid(
            self.gate_logit
        )

        out = x + gate * message
        # [B,C,W,D,P]

        with torch.no_grad():
            self.last_gate = gate.detach()
            self.last_expert_mass = expert_mass.detach()

        return out

    @torch.no_grad()
    def build_adjacency(self, router_probs):
        """
        Optional analysis helper.

        Materializes the patch-wise channel graph only when explicitly called:
            A: [B,W,C,C]

        A_ij = sum_e R_ie R_je / sum_c R_ce

        Rows sum to approximately 1 because each router row is a probability
        distribution. Avoid calling this routinely for very high-C datasets.
        """

        r = router_probs
        expert_mass = r.sum(dim=1).clamp_min(self.eps)  # [B,W,E]
        inv_mass = 1.0 / expert_mass

        adjacency = torch.einsum(
            "biwe,bjwe,bwe->bwij",
            r,
            r,
            inv_mass,
        )

        return adjacency



class MoEInceptionBlock(nn.Module):
    """
    x: [B, C, W, D, P]  ->  [B, C, W, D, P]
    Each layer has its own experts (different parameters).
    """
    def __init__(self, d_model, num_experts, num_kernels=7,
                 k_freq=1, temperature=0.8, use_sparse_routing=True,
                 dropout=0.0):
        super().__init__()
        self.num_experts = num_experts

        self.router = HybridRouter(num_experts=num_experts, d_model=d_model, k_freq=k_freq, temperature=temperature)
        self.use_sparse_routing = use_sparse_routing
        self.sparse_routing = RoutingSparsifier(num_experts) if use_sparse_routing else None

        self.experts = nn.ModuleList([
            InceptionExpert(d_model, num_kernels=num_kernels)
            for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_router=False):
        B, C, W, D, P = x.shape
        residual = x

        logits = self.router(x)  # [B,C,W,E]

        # Keep the dense soft probabilities for the channel graph.
        # These remain differentiable even when RoutingSparsifier is enabled.
        soft_weights = F.softmax(
            logits / self.router.temperature,
            dim=-1,
        )  # [B,C,W,E]

        expert_weights = soft_weights
        
        expert_usage = soft_weights.mean(
            dim=(0, 1, 2)
        )
        
        with torch.no_grad():
            self.last_expert_usage = expert_usage.detach()


        if self.use_sparse_routing:
            expert_weights = self.sparse_routing(
                expert_weights
            )  # [B,C,W,E]

        # expert forward
        x_flat = x.reshape(B * C * W, D, P)  # [BCW,D,P]
        expert_outs = torch.stack(
            [e(x_flat) for e in self.experts],
            dim=1,
        )  # [BCW,E,D,P]
        expert_outs = expert_outs.reshape(
            B, C, W, self.num_experts, D, P
        )

        mix_weights = expert_weights.unsqueeze(-1).unsqueeze(-1)
        # [B,C,W,E,1,1]

        out = (
            expert_outs * mix_weights
        ).sum(dim=3)  # [B,C,W,D,P]

        out = self.dropout(out)
        out = out + residual

        # LayerNorm over D
        out = out.permute(0, 1, 2, 4, 3)  # [B,C,W,P,D]
        out = self.norm(out)
        out = out.permute(0, 1, 2, 4, 3)  # [B,C,W,D,P]

        if return_router:
            return out, soft_weights

        return out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.num_channels = configs.enc_in
        self.num_class = configs.num_class
        self.d_model = getattr(configs, "d_model",  64)
        self.num_experts = min(10, max(3, int(math.floor(math.log2(self.num_channels)) * 2)))
        self.num_moe_layers = getattr(configs, "num_moe_layers", 2) 
        self.k_freq = getattr(configs, "k_freq", 1)
        self.router_temperature = getattr(configs, "router_temperature", 0.8)
        self.use_sparse_routing = getattr(configs, "use_sparse_routing", True)
        if self.use_sparse_routing:
            self.top_k = "sparse"
        else:
            self.top_k = "dense"
        self.moe_dropout = getattr(configs, "moe_dropout", 0.0)
        self.num_kernels = getattr(configs, "num_kernels", 7)
        self.period_flag = getattr(configs, "period_flag", 0)

        self.use_router_graph = getattr(configs, "use_router_graph", True)
        self.graph_dropout = getattr(configs, "graph_dropout", 0.0)
        self.graph_init_gate = getattr(configs, "graph_init_gate", 0.1)
        self.graph_detach_router = getattr(
            configs,
            "graph_detach_router",
            False,
        )

        dataset_name = configs.model_id
        file_path = "ICLR2027/uea_top5_periods_sgn_wo_duplicates.txt" Location
        PERIOD_MAP, CANDIDATE_MAP = load_period_map(file_path)
        self.period = get_period(dataset_name, PERIOD_MAP, CANDIDATE_MAP, self.period_flag)
        print("Period: ", self.period)
        self.patch_overlap = float(getattr(configs, "patch_overlap", 0))
        self.patch_stride = int(getattr(configs, "patch_stride", 0))

        if not 0.0 <= self.patch_overlap < 1.0:
            raise ValueError(
                "patch_overlap must be in [0,1), "
                f"received {self.patch_overlap}"
            )
        
        self.use_fod = getattr(configs, "use_fod", True)
        embedding_channels = (2 if self.use_fod else 1)
        self.embedding = nn.Conv1d(embedding_channels, self.d_model, 3, padding=1, bias=False)
        
        max_pos_len = max(5000, self.period)
        self.pos_emb = PositionalEmbedding(self.d_model, max_len=max_pos_len)
        self.patch_pos_emb = PositionalEmbedding(self.d_model)
        
        self.moe_layers = nn.ModuleList([
            MoEInceptionBlock(
                d_model=self.d_model,
                num_experts=self.num_experts,
                num_kernels=self.num_kernels,
                k_freq=self.k_freq,
                temperature=self.router_temperature,
                use_sparse_routing=self.use_sparse_routing,
                dropout=self.moe_dropout,
            )
            for _ in range(self.num_moe_layers)
        ])

        # Patch-wise channel graph induced by the FINAL MoE layer's
        # soft routing distribution. It mixes C while preserving W and P.
        self.channel_graph = RouterInducedChannelGraph(
            d_model=self.d_model,
            dropout=self.graph_dropout,
            init_gate=self.graph_init_gate,
            detach_router=self.graph_detach_router,
        )

        self.seq_layer = InterPatchSequenceLayer(d_model=self.d_model)
        
        # Feature maps dimensionality reduction
        self.dim_red_kernel_size = getattr(configs,"dim_red_kernel_size", 5)
        
        if self.dim_red_kernel_size == 1:
            self.dim_red_kernel_size = self.period
            
        self.dim_redux_conv = nn.Sequential(nn.Conv1d(self.num_channels * self.d_model, self.d_model, self.dim_red_kernel_size))
        
        self.res_proj = nn.Conv1d(self.num_channels * self.d_model, self.d_model, 1) # Dimensionality reduction residual
        
        self.out_feat_dim = self.d_model * 2
        self.norm = nn.LayerNorm(self.out_feat_dim)
        self.head = nn.Linear(self.out_feat_dim, self.num_class)
                
        
    def _get_patch_stride(self, patch_size):
        """
        Returns the integer stride used for patch extraction.

        patch_stride > 0 takes priority over patch_overlap.
        """

        if self.patch_stride > 0:
            stride = self.patch_stride
        else:
            stride = int(
                math.floor(
                    patch_size
                    * (1.0 - self.patch_overlap)
                    + 0.5
                )
            )

        stride = max(1, min(stride, patch_size))

        return stride
    
    
    
    def forward_single(self, x, P, seq_layer):
        """
        x: [B, L, C]

        Returns:
            [B, C, D, L]

        Required:
            self.embedding must use in_channels=2 if using First Order Difference
            self.patch_overlap must be in [0.0, 1.0)
        """

        # [B,L,C] -> [B,C,L]
        x = x.permute(0, 2, 1).contiguous()

        B, C, L = x.shape
        original_length = L

        if self.use_fod:
            difference = torch.diff(
                x,
                dim=-1,
                prepend=x[..., :1],
            )  # [B,C,L]
        else:
            difference = None

        
        overlap = self.patch_overlap

        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                f"patch_overlap must be in [0,1), received {overlap}"
            )

        # Optional explicit stride overrides overlap percentage.
        explicit_stride = int(
            getattr(self, "patch_stride", 0)
        )

        if explicit_stride > 0:
            stride = explicit_stride
        else:
            stride = int(
                P * (1.0 - overlap) + 0.5
            )

        stride = max(1, min(stride, P))

        if L <= P:
            W = 1
        else:
            W = math.ceil(
                (L - P) / stride
            ) + 1

        padded_length = (
            (W - 1) * stride + P
        )

        pad_length = padded_length - L

        if pad_length > 0:
            x = F.pad(
                x,
                (0, pad_length),
                mode="constant",
                value=0.0,
            )
            if self.use_fod:
                difference = F.pad(
                    difference,
                    (0, pad_length),
                    mode="constant",
                    value=0.0,
                )
        
        raw_patches = x.unfold(
            dimension=-1,
            size=P,
            step=stride,
        )  # [B,C,W,P]

        W = raw_patches.size(2)


        if self.use_fod:

            difference_patches = difference.unfold(
                dimension=-1,
                size=P,
                step=stride,
            )  # [B,C,W,P]

            patch_input = torch.stack(
                [
                    raw_patches,
                    difference_patches,
                ],
                dim=3,
            )
            # [B,C,W,2,P]

            x_flat = patch_input.contiguous().reshape(
                B * C * W,
                2,
                P,
            )
            # [BCW,2,P]

        else:

            patch_input = raw_patches.unsqueeze(
                3
            )
            # [B,C,W,1,P]

            x_flat = patch_input.contiguous().reshape(
                B * C * W,
                1,
                P,
            )
            # [BCW,1,P]
        
        
        x = self.embedding(
            x_flat
        )  # [BCW,D,P]

        # Within-patch positional embedding
        x = x.permute(
            0, 2, 1
        )  # [BCW,P,D]

        x = x + self.pos_emb(x)

        x = x.permute(
            0, 2, 1
        )  # [BCW,D,P]

        x = x.reshape(
            B,
            C,
            W,
            self.d_model,
            P,
        )  # [B,C,W,D,P]

        if W > self.patch_pos_emb.pe.size(1):
            raise ValueError(
                f"Number of patches W={W} exceeds positional "
                f"encoding limit {self.patch_pos_emb.pe.size(1)}"
            )

        patch_pos = (
            self.patch_pos_emb.pe[:, :W]
            .unsqueeze(1)
            .unsqueeze(-1)
        )  # [1,1,W,D,1]

        x = x + patch_pos

        out = x
        router_probs = None

        # Use the final MoE layer's soft routing distribution to induce
        # the patch-wise channel graph. Expert identities are layer-specific,
        # so router distributions from different MoE layers are not averaged.
        for layer_idx, moe in enumerate(self.moe_layers):
            if layer_idx == len(self.moe_layers) - 1:
                out, router_probs = moe(
                    out,
                    return_router=True,
                )
            else:
                out = moe(out)

        # MoE:       local processing over P
        # Graph:     cross-channel interaction over C, at each patch W
        # InterPatch: temporal interaction over W, independently per channel
        if self.use_router_graph:
            out = self.channel_graph(
                out,
                router_probs,
            )  # [B,C,W,D,P]

        # Inter-patch sequence modelling
        out = seq_layer(
            out
        )  # [B,C,W,D,P]

        # Convert to the format expected by F.fold:
        #
        # [B,C,W,D,P]
        # -> [B,C,D,P,W]
        # -> [B*C,D*P,W]
        patch_columns = out.permute(
            0,
            1,
            3,
            4,
            2,
        ).contiguous()

        patch_columns = patch_columns.reshape(
            B * C,
            self.d_model * P,
            W,
        )  # [BC,D*P,W]

        reconstructed = F.fold(
            patch_columns,
            output_size=(1, padded_length),
            kernel_size=(1, P),
            stride=(1, stride),
        )  # [BC,D,1,padded_length]

        reconstructed = reconstructed.squeeze(
            2
        )  # [BC,D,padded_length]

        overlap_columns = torch.ones(
            B * C,
            P,
            W,
            dtype=out.dtype,
            device=out.device,
        )

        overlap_count = F.fold(
            overlap_columns,
            output_size=(1, padded_length),
            kernel_size=(1, P),
            stride=(1, stride),
        )  # [BC,1,1,padded_length]

        overlap_count = overlap_count.squeeze(
            2
        )  # [BC,1,padded_length]

        # Average positions that receive contributions
        # from multiple overlapping patches.
        reconstructed = reconstructed / overlap_count.clamp_min(
            1.0
        )

        reconstructed = reconstructed.reshape(
            B,
            C,
            self.d_model,
            padded_length,
        )
        reconstructed = reconstructed[
            :,
            :,
            :,
            :original_length,
        ]  # [B,C,D,L]

        return reconstructed    


    # =====================================================
    def forward(self, x_enc, *args):
        """
        x_enc: [B, L, C]
        """
        out = self.forward_single(x_enc, self.period, self.seq_layer)
        # [B, C, D, L] -> [B, C*D, L]
        B, C, D, L = out.shape
        res = out.reshape(B, C * D, L)
        out = self.dim_redux_conv(res)  # [B, D, L'] depending on kernel
        out = out + self.res_proj(res[:, :, :out.shape[-1]])
        avg_feat = out.mean(dim=-1) # [B,D]
        max_feat = out.max(dim=-1).values   # [B,D]
        out = torch.cat([max_feat, avg_feat], dim=1)
        out = self.norm(out)
        out = self.head(out)        # [B, num_class]
        return out