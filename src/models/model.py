import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B,C,D',H',W']
        B, C, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B,N,C]
        return tokens, (D, H, W)

class TokenMixerBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = self.dwconv(h.transpose(1,2)).transpose(1,2)
        x = x + self.drop(h)
        h2 = self.norm2(x)
        h2 = self.fc2(F.gelu(self.fc1(h2)))
        x = x + self.drop(h2)
        return x

class TokenDownsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, grid_shape):
        B, N, C = x.shape
        D, H, W = grid_shape
        x3 = x.transpose(1,2).reshape(B, C, D, H, W)
        x3 = F.avg_pool3d(x3, kernel_size=2, stride=2)
        D2, H2, W2 = x3.shape[-3:]
        x2 = x3.flatten(2).transpose(1,2)
        x2 = self.proj(x2)
        return x2, (D2, H2, W2)

class HierarchicalEncoder(nn.Module):
    def __init__(self, dim: int, depth_L: int, blocks_per_stage, dropout: float):
        super().__init__()
        assert depth_L == len(blocks_per_stage)
        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth_L):
            self.stages.append(nn.Sequential(*[TokenMixerBlock(dim, dropout) for _ in range(blocks_per_stage[i])]))
            self.down.append(TokenDownsample(dim) if i < depth_L - 1 else None)

    def forward(self, x, grid_shape):
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x, grid_shape = self.down[i](x, grid_shape)
        return x

class RefinementModule(nn.Module):
    def __init__(self, dim: int, reduction: int, zscore_on_refined: bool):
        super().__init__()
        self.zscore_on_refined = zscore_on_refined
        hidden = max(1, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        pooled = x.mean(dim=1)  # [B,C]
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        x = x * gate[:, None, :]
        if self.zscore_on_refined:
            mu = x.mean(dim=(1,2), keepdim=True)
            std = x.std(dim=(1,2), keepdim=True) + 1e-6
            x = (x - mu) / std
        return x

class ClassifierHead(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        pooled = x.mean(dim=1)
        h = F.gelu(self.fc1(pooled))
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)

class ParkinsonFoundationNet(nn.Module):
    def __init__(self, in_ch: int, patch_size: int, embed_dim: int, depth_L: int, blocks_per_stage,
                 encoder_dropout: float, refinement_reduction: int, refinement_zscore: bool,
                 clf_hidden: int, clf_dropout: float):
        super().__init__()
        self.patch = PatchEmbedding3D(in_ch, embed_dim, patch_size)
        self.encoder = HierarchicalEncoder(embed_dim, depth_L, blocks_per_stage, encoder_dropout)
        self.refine = RefinementModule(embed_dim, refinement_reduction, refinement_zscore)
        self.head = ClassifierHead(embed_dim, clf_hidden, clf_dropout)

    def forward(self, x):
        tokens, grid = self.patch(x)
        tokens = self.encoder(tokens, grid)
        tokens = self.refine(tokens)
        logit = self.head(tokens)
        return logit
