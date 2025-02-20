# Modified from the original implementation: Copyright 2021 Yuan Gong, Yu-An Chung, James Glass
# Repository: https://github.com/YuanGongND/ast
# 
# Licensed under the BSD 3-Clause License.
# This code has been adapted and modified for this project.
# The original implementation can be found in the Audio Spectrogram Transformer (AST) repository.
# 
# This software is provided "AS IS", without warranties or conditions of any kind.
# See the License for specific terms and conditions.

import os

os.environ["TORCH_HOME"] = "./pretrained_models"
import random
import torch
import torch.nn as nn
import timm
from typing import Optional
from timm.layers import to_2tuple, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block, LayerScale
from .pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        PatchEmbed module for image or audio input.

        Args:
            img_size (int or tuple): Size of the input image or audio.
            patch_size (int or tuple): Size of each patch.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the output embeddings.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,  # Ensure Mlp class is defined or imported appropriately
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a1 = norm_layer(dim)
        self.norm1_a2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,  # Assuming Attention class is updated to use this
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.norm2_a1 = norm_layer(dim)
        self.norm2_a2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        if modality is None:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        elif modality == "a1":
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1_a1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_a1(x))))
        elif modality == "a2":
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1_a2(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_a2(x))))
        return x


# the finetuned CAV-MAE model
class ASTEncoder(nn.Module):
    def __init__(
        self,
        audio_length=256,
        mel_bins=512,
        patch_size=16,
        img_size=224,
        embed_dim=768,
        proj_dim=512,
        modality_specific_depth=11,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        tr_pos=True,
    ):
        super().__init__()
        timm.models.vision_transformer.Block = Block

        self.audio_length = audio_length
        self.mel_bins = mel_bins

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a1 = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_a2 = PatchEmbed(img_size, patch_size, 1, embed_dim)

        self.patch_embed_a1.num_patches = int(
            self.audio_length * self.mel_bins / 256
        )  # 128/256
        self.patch_embed_a2.num_patches = int(self.audio_length * self.mel_bins / 256)
        print(
            "Number of Audio Patches: {:d}, Number of Score Audio Patches: {:d}".format(
                self.patch_embed_a1.num_patches,
                self.patch_embed_a2.num_patches,
            )
        )

        self.modality_a1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_a2 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a1 = nn.Parameter(
            torch.zeros(1, self.patch_embed_a1.num_patches, embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding
        self.pos_embed_a2 = nn.Parameter(
            torch.zeros(1, self.patch_embed_a2.num_patches, embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding

        self.blocks_a1 = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(modality_specific_depth)
            ]
        )
        self.blocks_a2 = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(modality_specific_depth)
            ]
        )
        self.blocks_u = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(12 - modality_specific_depth)
            ]
        )

        self.norm_a1 = norm_layer(embed_dim)
        self.norm_a2 = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)
        hidden_dim = int(embed_dim * 2)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, label_dim),
        # )
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, 512),  # First additional linear layer
        #     nn.ReLU(),  # Non-linearity (ReLU activation function)
        #     nn.Dropout(0.5),  # Dropout for regularization
        #     nn.Linear(512, 256),  # Second additional linear layer
        #     nn.ReLU(),  # Another ReLU activation function
        #     nn.Dropout(0.3),  # Another dropout layer
        #     nn.Linear(
        #         256, label_dim
        #     ),  # Final linear layer outputting the label dimension
        # )

        # Add learnable scaling parameters
        # self.output_scale = nn.Parameter(torch.ones(1))
        # self.output_shift = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.initialize_weights()
        # Store MLP layer names

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)

        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a1 = get_2d_sincos_pos_embed(
            self.pos_embed_a1.shape[-1],
            8,
            int(self.patch_embed_a1.num_patches / 8),
            cls_token=False,
        )
        self.pos_embed_a1.data.copy_(
            torch.from_numpy(pos_embed_a1).float().unsqueeze(0)
        )

        pos_embed_a2 = get_2d_sincos_pos_embed(
            self.pos_embed_a2.shape[-1],
            8,
            int(self.patch_embed_a2.num_patches / 8),
            cls_token=False,
        )
        self.pos_embed_a2.data.copy_(
            torch.from_numpy(pos_embed_a2).float().unsqueeze(0)
        )

        w = self.patch_embed_a1.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_a2.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a1, std=0.02)
        torch.nn.init.normal_(self.modality_a2, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a1, a2):

        a1 = a1.unsqueeze(1)
        a1 = a1.transpose(2, 3)
        a1 = self.patch_embed_a1(a1)
        a1 = a1 + self.pos_embed_a1
        a1 = a1 + self.modality_a1

        a2 = a2.unsqueeze(1)
        a2 = a2.transpose(2, 3)
        a2 = self.patch_embed_a2(a2)
        a2 = a2 + self.pos_embed_a2
        a2 = a2 + self.modality_a2

        for blk in self.blocks_a1:
            a1 = blk(a1)

        for blk in self.blocks_a2:
            a2 = blk(a2)

        x = torch.cat((a1, a2), dim=1)

        for blk in self.blocks_u:
            x = blk(x)

        x = self.norm(x)

        x = self.proj(x)

        # x = x.mean(
        #     dim=1
        # )  # mean pooling over patches. This is the mean-pooled representation of the audio
        # x = self.mlp_head(x)
        # we return raw encoder output to cross-attention module

        return x
