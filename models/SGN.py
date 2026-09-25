import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_1D
import numpy as np

class PeriodMerging(nn.Module):
    def __init__(self, dim, group):
        super().__init__()

        self.group = group
        self.d_model = dim
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(2 * dim, dim)

    def forward(self, x):
        B, C, num, T= x.shape
        x = x.reshape(B, self.group, self.d_model, num, T)
        x = x.reshape(B*self.group, self.d_model, num, T)

        last_window = None
        if num <= 4:
            merged_windows = []
            for i in range(num - 1):
                merged_window = torch.cat([x[:, :, i, :], x[:, :, i+1, :]], dim=1)
                merged_windows.append(merged_window)

            x_merged = torch.stack(merged_windows, dim=2)
            x_merged = x_merged.reshape(B*self.group, 2*self.d_model, -1)

        else:
            if num % 2 == 1:
                last_window = x[:, :, -1, :]
                x = x[:, :, :-1, :]
                num -= 1

            x_even = x[:, :, 0::2, :]
            x_odd = x[:, :, 1::2, :]

            x_merged = torch.cat([x_even, x_odd], dim=1)  # (B, 2C, num, T)


            x_merged = x_merged.view(B*self.group, 2*self.d_model, -1)


        x_merged = x_merged.permute(0, 2, 1)
        x_reduced = self.reduction(x_merged)
        x_reduced = x_reduced.permute(0, 2, 1)

        x_reduced = x_reduced.reshape(B*self.group, self.d_model, -1, T)

        if last_window is not None:
            last_window = last_window.unsqueeze(2)
            x_reduced = torch.cat([x_reduced, last_window], dim=2)

        x_reduced = x_reduced.reshape(B*self.group, self.d_model, -1)


        x_reduced = x_reduced.permute(0, 2, 1)
        x_normalized = self.norm(x_reduced)
        x_normalized = x_normalized.permute(0, 2, 1)
        x_normalized = x_normalized.reshape(B*self.group, self.d_model, -1, T)
        x_normalized = x_normalized.reshape(B, self.group*self.d_model, -1, T)

        return x_normalized


class DilatedConvBlock(nn.Module):
    def __init__(self, group, d_model, ratio, num_kernel=5, dropout=0.1):
        super().__init__()

        self.group = group
        self.d_model = d_model

        self.conv = nn.Sequential(
            Inception_Block_1D(group*d_model, group * d_model * ratio, group=group*d_model, num_kernels=num_kernel),
            nn.GELU(),
            Inception_Block_1D(group * d_model * ratio, group * d_model, group=group*d_model, num_kernels=num_kernel)
        )
        self.norm = nn.BatchNorm1d(d_model)

        self.ffn1pw1 = nn.Conv1d(in_channels=group * d_model, out_channels=group * d_model * ratio, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=group)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=group * d_model * ratio, out_channels=group * d_model, kernel_size=1,stride=1,
                                 padding=0, dilation=1, groups=group)
        self.ffn1drop1 = nn.Dropout(dropout)
        self.ffn1drop2 = nn.Dropout(dropout)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=group * d_model, out_channels=group * d_model * ratio, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=group * d_model * ratio, out_channels=group * d_model, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2drop2 = nn.Dropout(dropout)

    def forward(self, x):

        B, C, num, T = x.shape
        input = x

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * num, C, T)
        x = self.conv(x)
        x = x.reshape(B*num, self.group, self.d_model, T)
        x = x.reshape(B*num*self.group, self.d_model, T)
        x = self.norm(x)
        x = x.reshape(B * num, self.group, self.d_model, T)
        x = x.reshape(B * num, self.group*self.d_model, T)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B * num, self.group, self.d_model, T)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * num, self.d_model * self.group, T)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B * num, self.d_model, self.group, T)
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(B * num, self.group, self.d_model, T)
        x = x.reshape(B*num, C, T)
        x = x.reshape(B, num, C, T)
        x = x.permute(0, 2, 1, 3)

        return x + input


class DilatedConvEncoder(nn.Module):
    def __init__(self, block_num, d_model, ratio_ffn ,num_kernels, group,dropout):
        super().__init__()
        self.net = nn.Sequential(
            *[
            DilatedConvBlock(
                    # kernel_size=kernel_size,
                    group=group,
                    d_model= d_model,
                    ratio= ratio_ffn,
                    num_kernel=num_kernels,
                    dropout=dropout,
                )
            for i in range(block_num)
            ]
        )


    def forward(self, x):
        return self.net(x)


class TemporalSwinTransformerBlock(nn.Module):
    def __init__(self, shift_size, attn_layer=None):
        super().__init__()
        self.tcn = attn_layer
        self.shift_size = shift_size

    def forward(self, x):
        B, C, num, T = x.shape

        x = x.reshape(B, C, num*T)

        # cycle shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-T // 2,), dims=(2,))
        else:
            shifted_x = x


        x_windows = shifted_x.reshape(B, C, num, T)  # B, C, num, T

        x_windows = self.tcn(x_windows)

        shifted_x = x_windows.reshape(B, C, num*T)

        # cycle shift
        if self.shift_size > 0:
            x_new = torch.roll(shifted_x, shifts=(T // 2,), dims=(2,))
            x_new[:, :, :T // 2] = x[:, :, :T // 2]
        else:
            x_new = shifted_x

        x_new = x_new.reshape(B, C, num, T)

        return x_new

class TemporalBasicLayer(nn.Module):
    def __init__(self, dim, depth, group,attn_layer=None, downsample=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            TemporalSwinTransformerBlock(
                shift_size = i % 2,
                attn_layer=attn_layer,
            ) for i in range(depth)
        ])

        # downsample
        if downsample is not None:
            self.downsample = downsample(dim=dim, group=group )
        else:
            self.downsample = None


    def forward(self, x):
        """
        x: Tensor of shape (B, T, C)
        """
        for blk in self.blocks:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class GumbelGroupEmbedding(nn.Module):
    def __init__(self, C, d_model, max_groups=8, temperature_init=1.0, temperature_min=0.1, anneal_rate=0.0003, sim_loss_weight=0.1):
        super().__init__()
        self.C = C
        self.G = max_groups
        self.d_model = d_model
        self.sim_loss_weight = sim_loss_weight


        # if self.config.data == "UCIHAR":
        # group_matrix_path = "/shared/projects/sakib/GitHub/Multi-variate-TSC-NIPS/sgn_extras/group_matrix/groups_matrix_U_6.npy"
        # # FLAAP
        group_matrix_path = "/shared/projects/sakib/GitHub/Multi-variate-TSC-NIPS/sgn_extras/group_matrix/groups_matrix_F_5.npy"
        
        groups_matrix = torch.tensor(np.load(group_matrix_path), dtype=torch.float32)
        self.assign_logits = nn.Parameter(groups_matrix)
        
        # The one line below this line is for random initialization of groups
        # self.assign_logits = nn.Parameter(torch.randn(C, max_groups) * 0.02) # random initialization of groups
        
        self.group_embeds = DataEmbedding(1, self.d_model)
        # Gumbel Softmax
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.anneal_rate = anneal_rate

        self.register_buffer("temperature", torch.tensor(temperature_init,dtype=torch.float32))
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))


    def gumbel_softmax(self, logits, hard=False):
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            y = (logits + gumbel_noise) / self.temperature
        else:
            y = logits / self.temperature

        y = F.softmax(y, dim=-1)
        if hard:
            y_hard = F.one_hot(y.argmax(dim=-1), num_classes=self.G).float()
            return (y_hard - y).detach() + y
        return y

    def update_temperature(self):
        self.training_step += 1
        new_temp = self.temperature_init * np.exp(-self.anneal_rate * self.training_step.cpu().item())
        self.temperature = torch.tensor(max(self.temperature_min, new_temp), dtype=torch.float32)

    def compute_similarity_matrix(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        x = x.permute(2, 0, 1).reshape(C, B * T)  # (C, B*T)
        x = F.normalize(x, dim=1)
        sim = torch.matmul(x, x.t())  # (C, C)
        return sim

    def similarity_regularization(self, assign_probs, sim_matrix):
        sim_matrix = (sim_matrix + 1) / 2  # [-1, 1] → [0, 1]
        diff = assign_probs.unsqueeze(1) - assign_probs.unsqueeze(0)  # (C, C, G)
        loss = (sim_matrix.unsqueeze(-1) * diff.pow(2)).sum() / (self.C * self.C)
        return loss

    def forward(self, x, return_info=False, hard_assign=False):
        """
        x: (B, T, C)
        """
        if not self.training:

            self.temperature = torch.tensor(self.temperature_min, dtype=torch.float32)
            hard_assign = True
        else:
            self.update_temperature()


        B, T, C = x.shape
        assign_probs = self.gumbel_softmax(self.assign_logits, hard=hard_assign)  # (C, G)

        x_grouped = torch.einsum('btc,cg->btg', x, assign_probs)

        # embedding (B, T, 1) → (B, T, d_model)
        x_grouped = x_grouped.permute(0, 2, 1).unsqueeze(-1)  # (B, G, T, 1)
        x_grouped = x_grouped.reshape(B * self.G, T, 1)
        x_grouped = self.group_embeds(x_grouped, None)  # (B, G, T, d_model)
        x_grouped = x_grouped.reshape(B, self.G, T, self.d_model)

        x_out = x_grouped.permute(0, 1, 3, 2).reshape(B, self.G * self.d_model, T)

        sim_loss = None
        if self.training and self.sim_loss_weight > 0:
            sim_matrix = self.compute_similarity_matrix(x.detach())  # 避免梯度回传
            sim_loss = self.similarity_regularization(assign_probs, sim_matrix)

        if return_info:
            group_usage = assign_probs.mean(dim=0).detach().cpu().numpy()
            group_id = assign_probs.argmax(dim=1).detach().cpu().numpy()
            return {
                "out": x_out,
                "assign_probs": assign_probs.detach(),
                "group_id": group_id,
                "group_usage": group_usage
            }
        return x_out, sim_loss



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.num_classes = configs.num_class
        self.num_layers = configs.e_layers
        self.period = configs.period
        self.depths = configs.depths
        self.seq_len = configs.seq_len
        self.group_num = configs.num_groups
        self.enc_embedding_group = GumbelGroupEmbedding(configs.enc_in, configs.d_model, max_groups=self.group_num)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = TemporalBasicLayer(
                dim=configs.d_model,
                depth=self.depths[i_layer],
                group=self.group_num,
                downsample=PeriodMerging if (i_layer < self.num_layers - 1) else None,
                attn_layer=DilatedConvEncoder(
                    block_num=configs.block_num,
                    # kernel_size=configs.kernel_size,
                    d_model = configs.d_model,
                    ratio_ffn = configs.mlp_ratio,
                    num_kernels=configs.num_kernels,
                    group=self.group_num,
                    dropout=configs.dropout
                )
            )

            self.layers.append(layer)

        self.norm = nn.LayerNorm(configs.d_model * self.group_num)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(configs.d_model * self.group_num, configs.num_class, bias=True) if configs.num_class > 0 else nn.Identity()



    def classification(self, x ,x_mark_enc):
        """
        x: (B, T, in_chans)
        """
        B, T, C = x.shape

        # embedding
        if self.group_num > 1:
            x, embeddings_for_loss = self.enc_embedding_group(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.enc_embedding(x,None)


        period = self.period

        # padding
        if self.seq_len % self.period != 0:
            length = ((self.seq_len // period) + 1) * period
            padding = torch.zeros([x.shape[0], length - self.seq_len, x.shape[2]]).to(x.device)
            out = torch.cat([x, padding], dim=1)

        else:
            length = self.seq_len
            out = x

        # reshape
        out = out.reshape(B, length // period, period,
                          x.shape[2]).permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            out = layer(out)     #B C num T


        out = out.reshape(B, out.shape[1], -1).permute(0, 2, 1)


        x = self.norm(out)  # B T C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x, embeddings_for_loss


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if(
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
