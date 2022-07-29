# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron commonly used in transformers. DROP(FC2(DROP(ACT(FC1(X)))))"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features # defaults to in_features given
        hidden_features = hidden_features or in_features # defaults to in_features given
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def masked_sin_pos_encoding(x, mask, num_pos_feats, temperature=10000, scale=2 * math.pi):
    """ Masked Sinusoidal Positional Encoding

    Parameters:
        x: [PATCH] tokens
        mask: the padding mask for [PATCH] tokens, 
            interpolation of the original mask in the NestedTensor obtained from fetching the sample from the dataset
        num_pos_feats: the size of channel dimension
        temperature: the temperature value
        scale: the normalization scale

    Returns:
        pos: Sinusoidal positional encodings
    """
    # x = [B, H/32 * W/32 + 100, 384], 
    # mask = [B, H/32, W/32], num_pos_feats = 384, temperature=10000, scale=2 * math.pi

    num_pos_feats = num_pos_feats // 2 # 192
    not_mask = ~mask

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3)

    return pos # t.s [B, H/32, W/32, 384]


def window_partition(x, window_size):
    """
    Parameters:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # "x.permute(0, 1, 3, 2, 4, 5)" -> (B, H // window_size, W // window_size, window_size, window_size, C)
    # ".view()" -> (B * H // window_size * W // window_size, window_size, window_size, C)
    # note that H // window_size = number of windows height-wise and 
    # W // window_size = number of windows width-wise
    # so H // window_size * W // window_size = total number of windows in the image
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows # t.s [B * total number of windows, window_size, window_size, C]


def window_reverse(windows, window_size, H, W):
    """
    Parameters:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ReconfiguredAttentionModule(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias -> extended with RAM.
    It supports both of shifted and non-shifted window.

    !!!!!!!!!!! IMPORTANT !!!!!!!!!!!
    The original attention module in Swin is replaced with the reconfigured attention module in Section 3.
    All the parameters are shared, so only the forward function is modified.
    See https://arxiv.org/pdf/2110.03921.pdf
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Parameters:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        # dim = 48 ; 48; 96 ; 96; 192 ; 192; 192; 192; 192 ; 192 ; 384 ; 384, 
        # window_size = tuple(7,7), 
        # num_heads = 8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads # 6 ; 6 ; 12 ; 12 ; 24 ; 24 ; 24 ; 24 ; 24 ; 24 ; 48 ; 48
        self.scale = qk_scale or head_dim ** -0.5 # 0.408 ; 0.408 ; 0.289 ; 0.289 ; 0.204 ; 0.204 ; 0.204 ; 0.204 ; 0.204 ; 0.204 ; 0.144 ; 0.144 

        # define a parameter table of relative position bias
        # the reason why it's 2x is because relative distance range is [-(window_size-1), window_size-1]
        # so the number of possible relative positions for just one side (height or width)
        # is window_size-1 - (-(window_size-1)) + 1 = 2 * window_size - 1
        # lower bound is the relative position of the first pixel in the window from the last pixel,
        # and the upper bound is the relative position of the last pixel in the window from the first pixel
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # t.s [13 * 13, 8] = [169, 8]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0]) # [0, 1, 2, 3, 4, 5, 6]
        coords_w = torch.arange(self.window_size[1]) # [0, 1, 2, 3, 4, 5, 6]
        # tensor([[[0, 0, 0, 0, 0, 0, 0],
        #          [1, 1, 1, 1, 1, 1, 1],
        #          [2, 2, 2, 2, 2, 2, 2],
        #          [3, 3, 3, 3, 3, 3, 3],
        #          [4, 4, 4, 4, 4, 4, 4],
        #          [5, 5, 5, 5, 5, 5, 5],
        #          [6, 6, 6, 6, 6, 6, 6]],

        #        [[0, 1, 2, 3, 4, 5, 6],
        #         [0, 1, 2, 3, 4, 5, 6],
        #         [0, 1, 2, 3, 4, 5, 6],
        #         [0, 1, 2, 3, 4, 5, 6],
        #         [0, 1, 2, 3, 4, 5, 6],
        #         [0, 1, 2, 3, 4, 5, 6],
        #         [0, 1, 2, 3, 4, 5, 6]]])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # t.s [2, 7, 7]
        # tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
        #          3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
        #          6],
        #         [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2,
        #          3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
        #          6]])
        coords_flatten = torch.flatten(coords, 1)  # t.s [2, 49]
        # get the relative distance from each coord to every other coord, resultant shape [2, 49, 49]
        # for ex. the first row is of shape [49] and is the distance between the first 0 in the first row,column, 
        # with every value in coords_flatten[1, :], 
        # tensor([[[ 0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2,
        #           -2, -2, -2, -2, -3, -3, -3, -3, -3, -3, -3, -4, -4, -4, -4, -4, -4,
        #           -4, -5, -5, -5, -5, -5, -5, -5, -6, -6, -6, -6, -6, -6, -6],
        # range of values is between [-6, 6]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # t.s [2, 49, 49]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # t.s [49, 49, 2]
        
        ## add 6 so that there are no -s, shift to start from 0
        ## range of values is now between [0, 12]
        relative_coords[:, :, 0] += self.window_size[0] - 1  
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # so first dim + second dim value is the actual index in the flattened img input
        # 2* because it has to capture both H and W distance, which are sel.window_size each
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # multiply first dim by 13
        # first row ex. [ 84,  83,  82,  81,  80,  79,  78,  71,  70,  69,  68,  67,  66,  65,
        #                 58,  57,  56,  55,  54,  53,  52,  45,  44,  43,  42,  41,  40,  39,
        #                 32,  31,  30,  29,  28,  27,  26,  19,  18,  17,  16,  15,  14,  13,
        #                  6,   5,   4,   3,   2,   1,   0],
        # second row ex.[ 85,  84,  83,  82,  81,  80,  79,  72,  71,  70,  69,  68,  67,  66,
        #                 59,  58,  57,  56,  55,  54,  53,  46,  45,  44,  43,  42,  41,  40,
        #                 33,  32,  31,  30,  29,  28,  27,  20,  19,  18,  17,  16,  15,  14,
        #                  7,   6,   5,   4,   3,   2,   1]
        # range of values is between [0, 168], this perfectly matches all indices in self.relative_position_bias_table
        relative_position_index = relative_coords.sum(-1)  # t.s [49, 49]
        # saved alongside params in state but is not update-able
        self.register_buffer("relative_position_index", relative_position_index)
        # (48, 144) x2
        # (96, 288) x2
        # (192, 576) x6
        # (384, 1152) x2
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # (48, 48) x2
        # (96, 96) x2
        # (192, 192) x6
        # (384, 384) x2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, det, mask=None, cross_attn=False, cross_attn_mask=None):
        """ Forward function.
        RAM module receives [Patch] and [DET] tokens and returns their calibrated ones

        Parameters:
            x: [PATCH] tokens
            det: [DET] tokens
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None -> mask for shifted window attention

            "additional inputs for RAM"
            cross_attn: whether to use cross-attention [det x patch] (for selective cross-attention)
            cross_attn_mask: mask for cross-attention

        Returns:
            patch_x: the calibrated [PATCH] tokens
            det_x: the calibrated [DET] tokens
        """
        # x = t.s [B, H/4, W/4, 48] ; [B, H/4, W/4, 96] ; ... ; (t.s. [B, H/32, W/32, 384], [B, H/32, W/32, 384])
        # det = t.s [B, 100, 48]; ... ; [B, 100, 384]
        # mask=None ; t.s [num_windows, 49, 49] ; ... ; None ; t.s [num_windows, 49, 49]
        # cross_attn=False ; ... ; True, 
        # cross_attn_mask=None ; ... ; t.s. [B, 1, 1, H_3 * W_3 + 100]

        assert self.window_size[0] == self.window_size[1]
        window_size = self.window_size[0]
        local_map_size = window_size * window_size

        # projection before window partitioning
        if not cross_attn:
            B, H, W, C = x.shape
            N = H * W # H/4 * W/4
            x = x.view(B, N, C) # t.s [B, H/4 * W/4, 48]
            x = torch.cat([x, det], dim=1) # t.s [B, H/4 * W/4 + 100, 48]
            # [B, H/4 * W/4 + 100, 144]
            full_qkv = self.qkv(x)
            # separate the PATCH from the DET tokens
            # patch_qkv has t.s. [B, H/4 * W/4, 144], det_qkv has shape [B, 100, 144]
            patch_qkv, det_qkv = full_qkv[:, :N, :], full_qkv[:, N:, :]
        else:
            B, H, W, C = x[0].shape
            N = H * W
            _, ori_H, ori_W, _ = x[1].shape
            ori_N = ori_H * ori_W

            shifted_x = x[0].view(B, N, C) # t.s [B, H/32 * W/32, 384]
            cross_x = x[1].view(B, ori_N, C) # t.s [B, H/32 * W/32, 384]
            x = torch.cat([shifted_x, cross_x, det], dim=1) # t.s [B, (H/32 * W/32) * 2 + 100, 384]
            full_qkv = self.qkv(x) # t.s [B, (H/32 * W/32) * 2 + 100, 1152]
            # separate the [PATCH]x[PATCH] qkv, the [PATCH]x[DET] qkv, and the [DET]x[DET] qkv
            # patch_qkv t.s [B, H/32 * W/32, 1152]
            # cross_patch_qkv t.s [B, H/32 * W/32, 1152]
            # det_qkv t.s [B, 100, 1152]
            patch_qkv, cross_patch_qkv, det_qkv = \
                full_qkv[:, :N, :], full_qkv[:, N:N + ori_N, :], full_qkv[:, N + ori_N:, :]
        patch_qkv = patch_qkv.view(B, H, W, -1) # t.s [B, H/4, W/4, 144] ; ... ; [B, H/32, W/32, 1152]

        # window partitioning for [PATCH] tokens
        patch_qkv = window_partition(patch_qkv, window_size)  # t.s [B * num_windows, 7,7,144] ; [B * num_windows, 7, 7, 1152]
        B_ = patch_qkv.shape[0] # B * num_windows
        # t.s [B * num_windows, 49, 3, 8, 6] ; ... ; [B * num_windows, 49, 3, 8, 48]
        patch_qkv = patch_qkv.reshape(B_, window_size * window_size, 3, self.num_heads, C // self.num_heads)
        _patch_qkv = patch_qkv.permute(2, 0, 3, 1, 4) # t.s [3, B * num_windows, 8, 49, 6] ;  ... ; [3, B*num_windows, 8, 49, 48]
        # first dimension indexes the window, second indexes the head, third indexes the pixel, fourth indexes the dimension
        patch_q, patch_k, patch_v = _patch_qkv[0], _patch_qkv[1], _patch_qkv[2] # each one has t.s [B * num_windows, 8, 49, 6] ; ... ; [B*num_windows, 8, 49, 48]

        # [PATCH x PATCH] self-attention using window partitions
        patch_q = patch_q * self.scale #  multiply by 0.408 ; 0.408 ; 0.289 ; 0.289 ; 0.204 ; 0.204 ; 0.204 ; 0.204 ; 0.204 ; 0.204 ; 0.144 ; 0.144 
        # matrix multiplication between qk to produce the attention matrix
        patch_attn = (patch_q @ patch_k.transpose(-2, -1)) # t.s [B * num_windows, 8, 49, 49] ; ... ; [B*num_windows, 8, 49, 49]
        # add relative pos bias for [patch x patch] self-attention
        # get pos bias for each of the pixels in the window from every other pixel in the window, this is why it's [49, 49]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # t.s [49, 49, 8] (where 8 is the num heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [8, 49, 49] = [nheads, window_num_values,window_num_values]
        # add the same position bias to all patch attention matrices for each window for all images in the batch
        patch_attn = patch_attn + relative_position_bias.unsqueeze(0) # t.s [B * num_windows, 8, 49, 49]

        # if shifted window is used, it needs to apply the mask
        if mask is not None:
            nW = mask.shape[0] # gets the number of windows
            # "view()" -> t.s [B, num_windows, 8, 49, 49]
            # "mask.unsqueeze(1).unsqueeze(0)" -> t.s [1, num_windows, 1, 49, 49]
            # the mask has the attention weightage between every value in the window with every other value,
            # weightage of -100 when the two values are in different "windows" due to the shift, weightage of 0 otherwise
            patch_attn = patch_attn.view(B_ // nW, nW, self.num_heads, local_map_size, local_map_size) + \
                         mask.unsqueeze(1).unsqueeze(0)
            patch_attn = patch_attn.view(-1, self.num_heads, local_map_size, local_map_size) # t.s [B*num_windows, 8, 49, 49]

        patch_attn = self.softmax(patch_attn)
        patch_attn = self.attn_drop(patch_attn)
        # "patch_attn @ patch_v" -> matmul, last two dimensions allow matmul ([49, 49] x [49, 6]) so result is of shape [B*num_windows, 8, 49, 6] ; ... ; [B*num_windows, 8, 49, 48]
        # "transpose()" -> [B*num_windows, 49, 8, 6] ; [B*num_windows, 49, 8, 48]
        patch_x = (patch_attn @ patch_v).transpose(1, 2).reshape(B_, window_size, window_size, C) # t.s [B *num_windows, 7, 7, 48] ; ... ; [B*num_windows, 7, 7, 384]

        """[DET] x [DET] self-attention"""
            
        # extract qkv for [DET] tokens
        det_qkv = det_qkv.view(B, -1, 3, self.num_heads, C // self.num_heads) # t.s [B, 100, 3, 8, 6] ; ... ; [B, 100, 3, 8, 48]
        det_qkv = det_qkv.permute(2, 0, 3, 1, 4) # t.s [3, B, 8, 100, 6] ; ... ; [3, B, 8, 100, 48]
        det_q, det_k, det_v = det_qkv[0], det_qkv[1], det_qkv[2] # each has t.s [B, 8, 100, 6] ; ... ; [B, 8, 100, 48]

        # if cross-attention is activated
        if cross_attn:

            # reconstruct the spatial form of [PATCH] tokens for global [DET x PATCH] attention
            cross_patch_qkv = cross_patch_qkv.view(B, ori_H, ori_W, 3, self.num_heads, C // self.num_heads) # [B, H/32, W/32, 3, 8, 48]
            patch_kv = cross_patch_qkv[:, :, :, 1:, :, :].permute(3, 0, 4, 1, 2, 5).contiguous() # [B, H/32, W/32, 2, 8, 48]
            patch_kv = patch_kv.view(2, B, self.num_heads, ori_H * ori_W, -1) # [2, B, 8, H/32 * W/32, 48]

            # extract "key and value" of [PATCH] tokens for cross-attention
            cross_patch_k, cross_patch_v = patch_kv[0], patch_kv[1] # each has t.s [B, 8, H/32 * W/32, 48]

            # bind key and value of [PATCH] and [DET] tokens for [DET X [PATCH, DET]] attention
            # det_k has t.s. [B, 8, H/32 * W/32 + 100, 48]
            # det_v has t.s. [B, 8, H/32 * W/32 + 100, 48]
            det_k, det_v = torch.cat([cross_patch_k, det_k], dim=2), torch.cat([cross_patch_v, det_v], dim=2)

        # [DET x DET] self-attention or binded [DET x [PATCH, DET]] attention
        det_q = det_q * self.scale
        # matmul between [DET] token queries and [DET] token patches
        # [B, 8, 100, 6] matmul  [B, 8, 6, 100] (last two dims are compatible, [100, 6] x [6, 100]) result shape [B, 8, 100, 100]
        # ; ... ; [B, 8, 100, 48] matmul [B, 8, 48, H/32 * W/32 + 100], result shape [B, 8, 100, H/32 * W/32 + 100]
        # attention matrix
        det_attn = (det_q @ det_k.transpose(-2, -1)) 
        # apply cross-attention mask if available
        if cross_attn_mask is not None:
            det_attn = det_attn + cross_attn_mask # [B, 8, 100, H/32 * W/32 + 100]
        det_attn = self.softmax(det_attn)
        det_attn = self.attn_drop(det_attn)
        # [B, 8, 100, 100] matmul [B, 8, 100, 6] = [B, 8, 100, 6]
        # ; ... ; [B, 8, 100, H/32 * W/32 + 100] matmul [B, 8, H/32 * W/32 + 100, 48] = [B, 8, 100, 48]
        det_x = (det_attn @ det_v).transpose(1, 2).reshape(B, -1, C) # t.s [B, 100, 48] ; ... ; [B, 100, 384]

        # reverse window for [PATCH] tokens <- the output of [PATCH x PATCH] self attention
        patch_x = window_reverse(patch_x, window_size, H, W) # t.s [B, H/4, W/4, 48] ; [B, H/32, W/32, 384]

        ## projection for outputs from multi-head
        # combine the [PATCH] and [DET] tokens again
        x = torch.cat([patch_x.view(B, H*W, C), det_x], dim=1) # t.s [B, H/4 * W/4 + 100, 48] ; ... ; [B, H/32 * W/32 + 100, 394]
        x = self.proj(x) # linearly project the tokens t.s [B, H/4 * W/4 + 100, 48] ; ... ; [B, H/32 * W/32 + 100, 384]
        x = self.proj_drop(x) 

        # decompose after FFN into [PATCH] and [DET] tokens
        # separate them again
        patch_x = x[:, :H * W, :].view(B, H, W, C)
        det_x = x[:, H * W:, :]

        return patch_x, det_x # t.s [B, H/4, W/4, 48], [B, 100, 48] ; ... ; [B, H/32, W/32, 384], [B, 100, 384]


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Parameters:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # dim = 48 ; 48; 96 ; 96; 192 ; 192; 192; 192; 192 ; 192 ; 384 ; 384 (follows depth = [2, 2, 6, 2], see BasicLayer), 
        # num_heads = 3 ; 3 ; 6 ; 6 ; 12 ; 12 ;12 ; 12 ; 12 ; 12;  24 ; 24, 
        # window_size=7, 
        # shift_size=0 ; 3 ; 0 ; 3 ; 0 ; 3 ; 0 ; 3 ; 0 ; 3 ; 0 ; 3
        # mlp_ratio=4., 
        # qkv_bias=True, 
        # qk_scale=None, 
        # drop=0., 
        # attn_drop=0., 
        # drop_path=0.,
        # act_layer=nn.GELU, 
        # norm_layer=nn.LayerNorm
        super().__init__()
        
        # additional instance variables added from finetune_det:
        # det_token_num = 100
        # det_pos_linear = nn.Linear(256, this block's self.dim)
        
        # additional variables added from BasicLayer's forward:
        # H = H/4 ; H/8, 
        # W = W/4 ; W/8
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = ReconfiguredAttentionModule(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # in features = 48 ; 48; 96 ; 96; 192 ; 192; 192; 192; 192 ; 192 ; 384 ; 384
        # hidden features = 192 ; 192 ; 384 ; 384 ; 768 ; 768 ; 768 ; 768 ; 768 ; 768 ; 1536 ; 1536
        # output features = 48 ; 48; 96 ; 96; 192 ; 192; 192; 192; 192 ; 192 ; 384 ; 384
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix, pos, cross_attn, cross_attn_mask):
        """ Forward function.

        Parameters:
            x: Input feature, tensor size (B, H*W + DET, C). i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.

            "additional inputs'
            pos: (patch_pos, det_pos)
            cross_attn: whether to use cross attn [det x [det + patch]]
            cross_attn_mask: attention mask for cross-attention

        Returns:
            x: calibrated & binded [PATCH, DET] tokens
        """
        # x = t.s [B, H/4 * W/4 + 100, 48] ; [B, H/8 * W/8 + 100, 96] ; ... ; [B, H/32 * W/32 + 100, 384]
        # mask_matrix = t.s [num_windows, 49, 49]
        # pos = (None, Parameter(t.s. [1, 100, 256])) ; ... ; 
        #           (t.s. [B, H/32, W/32, 384] masked sin positional embedding, Parameter(t.s. [1, 100, 256]))
        # cross_attn = False ; ... ; True
        # cross_attn_mask = None ; ... ; t.s. [B, 1, 1, H_3 * W_3 + 100]

        B, L, C = x.shape
        H, W = self.H, self.W # H/4, W/4 ; H/8, W/8 ; ... ; H/32, W/32

        assert L == H * W + self.det_token_num, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # x is only the image features, det is the 100 part, i.e. the query part
        x, det = x[:, :H * W, :], x[:, H * W:, :]
        x = x.view(B, H, W, C) # t.s [B, H/4, W/4, 48] ; [B, H/8, W/8, 96] ; ... ; [B, H/32, W/32, 384]
        orig_x = x

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        # pad last dimension by 0x0, i.e. unchanged
        # pad second last dimension (width) by pad_l=0 to the left and pad_r to the right
        # pad third last dimension (height) by pad_t=0 to the top and pad_b to the bottom
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # projection for det positional encodings: make the channel size suitable for the current layer
        # patch_pos = None ; t.s. [B, H/32, W/32, 384], 
        # det_pos = Parameter(t.s. [1, 100, 256])
        patch_pos, det_pos = pos
        # linearly project the position embedding parameters,
        det_pos = self.det_pos_linear(det_pos) # t.s [1, 100, 48] ; [1, 100, 96] ; ... ; [1, 100, 384]

        # cyclic shift
        if self.shift_size > 0: # False ; True ; False
            # roll shifts based on the given values and dimensions
            # if elements are shifted beyond the last position, they are wrapped around to the first position and vv.
            # shifts by -3 in dimension 1 and 2, i.e. in width and height dimensions, meaning shift left, wrap to the right, and shift up, wrap to the bottom
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix # t.s [num_windows, 49, 49]
        else:
            shifted_x = x
            attn_mask = None

        # prepare cross-attn and add positional encodings
        if cross_attn:
            # patch token (for cross-attention) + Sinusoidal pos encoding
            cross_patch = orig_x + patch_pos # t.s [B, H/32, W/32, 384]
            # det token + learnable pos encoding
            # add positional information to the detection tokens
            det = det + det_pos # t.s [B, 100, 384]
            # tuple (t.s. [B, H/32, W/32, 384], [B, H/32, W/32, 384])
            shifted_x = (shifted_x, cross_patch)
        else:
            # if cross_attn is deativated, only [PATCH] and [DET] self-attention are performed
            # det token + learnable pos encoding
            # add positional information to the detection tokens
            det = det + det_pos # t.s [B, 100, 48] ; [B, 100, 96]
            shifted_x = shifted_x

        # W-MSA/SW-MSA
        shifted_x, det = self.attn(shifted_x, mask=attn_mask,
                                   # additional parameters
                                   det=det,
                                   cross_attn=cross_attn,
                                   cross_attn_mask=cross_attn_mask) # returned t.s [B, H/4, W/4, 48], [B, 100, 48] ; ... ; [B, H/32, W/32, 384], [B, 100, 384]

        # reverse cyclic shift
        if self.shift_size > 0:
            # in dim 1 (width dimension), shift to the right by 3
            # in dim 2 (height dim), shift to the bottom by 3
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous() # take only non-padded/original values

        x = x.view(B, H * W, C) # t.s [B, H/4 * W/4, 48] ; ... ; [B, H/32 * W/32, 384]
        x = torch.cat([x, det], dim=1) # t.s [B, H/4 * W/4 + 100, 48] ; ... ; [B, H/32 * W/32 + 100, 384]

        ## FFN
        # residual connection
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # t.s [B, H/4 * W/4 + 100, 48] ; ... ; [B, H/32 * W/32 + 100, 384]

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Parameters:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, expand=True):
        # dim = 48 ; 96 ; 192 ; 384, 
        # norm_layer=nn.LayerNorm, 
        # expand=True ; True ; True ; False
        
        # additional instance variables introduced in finetune_det:
        # self.det_token_num = 100
        super().__init__()
        self.dim = dim

        # if expand is True, the channel size will be expanded, otherwise, return 256 size of channel
        expand_dim = 2 * dim if expand else 256 # 96 ; 192 ; 384 ; 256
        # in channels = 192 ; 384 ; 768 ; 1536
        # out channels = 96 ; 192 ; 384 ; 256
        self.reduction = nn.Linear(4 * dim, expand_dim, bias=False) 
        self.norm = norm_layer(4 * dim)

        # added for detection token [please ignore, not used for training]
        # not implemented yet.
        self.expansion = nn.Linear(dim, expand_dim, bias=False) # N/A
        self.norm2 = norm_layer(dim)

    def forward(self, x, H, W):
        """ Forward function.

        Parameters:
            x: Input feature, tensor size (B, H*W, C), i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.

        Returns:
            x: merged [PATCH, DET] tokens;
            only [PATCH] tokens are reduced in spatial dim, while [DET] tokens is fix-scale
        """
        # x = t.s [B, H/4 * W/4 + 100, 48],
        # H = H/4, W = W/4

        B, L, C = x.shape
        assert L == H * W + self.det_token_num, "input feature has wrong size"

        # separate the [PATCH] from the [DET] tokens
        x, det = x[:, :H * W, :], x[:, H * W:, :]
        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) # pad is any of the height and/or width is not even
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # take even x-values & even y-values
        x0 = x[:, 0::2, 0::2, :]  # B (H/4)/2 (W/4)/2 C = B H/8 W/8 C
        # take odd x-values & even y-values
        x1 = x[:, 1::2, 0::2, :]  # B H/8 W/8 C
        # take even x-values & odd y-values
        x2 = x[:, 0::2, 1::2, :]  # B H/8 W/8 C
        # take odd x-values & odd y-values
        x3 = x[:, 1::2, 1::2, :]  # B H/8 W/8 C
        # essentially move the values in the neighbouring region into the last dimension
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/8 W/8 4*C
        x = x.view(B, -1, 4 * C)  # B H/8*W/8 4*C = t.s [B, H/8 * W/8, 192]

        # simply repeating for DET tokens
        det = det.repeat(1, 1, 4) # t.s [B, 100, 48 * 4] = [B, 100, 192]

        x = torch.cat([x, det], dim=1) # t.s [B, H/8 * W/8 + 100, 192]
        x = self.norm(x)
        x = self.reduction(x) # t.s [B, H/8 * W/8 + 100, 96]

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Parameters:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 last=False,
                 use_checkpoint=False):
        # dim = 48 ; 96 ; 192 ; 384,
        # depth = 2 ; 2 ; 6 ; 2,
        # num_heads = 3 ; 6 ; 12 ; 24,
        # window_size=7,
        # mlp_ratio=4.,
        # qkv_bias=True,
        # qk_scale=None,
        # drop=0.,
        # attn_drop=0.,
        # drop_path=0.,
        # norm_layer=nn.LayerNorm,
        # downsample=PatchMerging module *reference*,
        # last=None ; None ; None; True,
        # use_checkpoint=False

        super().__init__()
        # additional instance variables added in finetune-det:
        # self.det_token_num = 100
        self.window_size = window_size
        self.shift_size = window_size // 2 # 3
        self.depth = depth
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # build blocks
        # MODULE LIST of size 2 ; 2 ; 6 ; 2
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, expand=(not last))
        else:
            self.downsample = None


    def forward(self, x, H, W, det_pos, input_mask, cross_attn=False):
        """ Forward function.

        Parameters:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            det_pos: pos encoding for det token
            input_mask: padding mask for inputs
            cross_attn: whether to use cross attn [det x [det + patch]]
        """
        # x = t.s [B, H/4 * W/4 + 100, 48] ; [B, H/8 * W/8 + 100, 96] ; ... ; [B, H/32 * W/32 + 100, 384]
        # H = H/4 ; ((H/4) + 1) // 2 ; ... ; ((H/16) + 1) // 2 hereafter referred to as H_3
        # W = W/4 ; ((W/4) + 1) // 2; ... ; ((W/16) + 1) // 2 hereafter referred to as W_3
        # det_pos = Parameter(t.s. [1, 100, 256]), 
        # input_mask = [B, H/32, W/32], 
        # cross_attn=False ; False ; ... ; True

        B = x.shape[0]

        # calculate attention mask for SW-MSA
        
        ## get the height and width such that it is divisible by 7, i.e. so that the image can be split into 7 windows
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # t.s [1, Hp, Wp, 1]
        
        
        h_slices = (slice(0, -self.window_size), # defines the first slice, taking 0 to -7
                    slice(-self.window_size, -self.shift_size), # defines the second slice, taking -7 to -3
                    slice(-self.shift_size, None)) # defines the third slice, taking -3 onwards
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # mask for cyclic shift
        mask_windows = window_partition(img_mask, self.window_size) # t.s [Hp/7 * Wp/7, 7, 7, 1] = [num_windows, 7, 7, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # t.s [num_windows, 49]
        # "mask_windows.unsqueeze(1)" -> t.s [num_windows, 1, 49], call this a
        # "mask_windows.unsqueeze(2)" -> t.s [num_windows, 49, 1], call this b
        # for window_num i and row j in a, subtract value x from b[i, j, 0] from each value in row j in a,
        # for ex. [6, 6, ..., 7, 8] - [6] = [0, 0, ..., 1, 2]
        # therefore the result is a tensor of shape [num_windows, 49, 49]
        # where for row i, the value in the i-th column is 0, everything else is either 0 or non-zero
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # fill the values which are non-zero with -100.0,
        # fill the values which are zeros with 0.0
        # mask indicates for each of the 49 values which ones it can pay attention to
        # the value will be 0 if both values are equal, i.e. 0 == 0, 1 == 1, 2 == 2, otherwise it will be -100
        # the reason for this is because the last window (width and heightwise) because of the shift, now consists of 4 columns/rows
        # which were originally next to each other, but the last 3 columns/rows were wrapped from the left of the image due to the shift
        # now since these two groups were originally not next to each other, we need to distinguish them
        # as the transformer might mistake them as being next to each other because transformers are permutation invariant
        # see img in temp/mask_example.png
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # compute sinusoidal pos encoding and cross-attn mask here to avoid redundant computation
        if cross_attn:

            _H, _W = input_mask.shape[1:] # H/32, W/32
            if not (_H == H and _W == W):
                # interpolate from existing values to make sure that the size of the input mask matches (H/32, W/32)
                input_mask = F.interpolate(input_mask[None].float(), size=(H, W)).to(torch.bool)[0]

            # sinusoidal pos encoding for [PATCH] tokens used in cross-attention
            patch_pos = masked_sin_pos_encoding(x, input_mask, self.dim) # t.s [B, H/32, W/32, 384]

            # attention padding mask due to the zero padding in inputs
            # the zero (padded) area is masked by 1.0 in 'input_mask'
            cross_attn_mask = input_mask.float()
            # don't pay attention to padded values so give attention weightage -100
            cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask != 0.0, float(-100.0)). \
                masked_fill(cross_attn_mask == 0.0, float(0.0))

            # pad for detection token (this padding is required to process the binded [PATCH, DET] attention
            cross_attn_mask = cross_attn_mask.view(B, H * W).unsqueeze(1).unsqueeze(2) # t.s [B, 1, 1, H_3 * W_3]
            # pad the last dimension (H_3 * W_3 dimension) to the left by 0, pad to the right by 100 with 0s
            cross_attn_mask = F.pad(cross_attn_mask, (0, self.det_token_num), value=0) # t.s [B, 1, 1, H_3 * W_3 + 100]

        else:
            patch_pos = None
            cross_attn_mask = None

        # zip pos encodings
        pos = (patch_pos, det_pos) # (None, Parameter(t.s. [1, 100, 256]))

        # first basic layer has 2 blocks ; second basic layer has 2 blocks
        for n_blk, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W

            # for selective cross-attention
            if cross_attn:
                _cross_attn = True
                _cross_attn_mask = cross_attn_mask
                _pos = pos # i.e., (patch_pos, det_pos)
            else:
                _cross_attn = False
                _cross_attn_mask = None
                _pos = (None, det_pos)

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask,
                                          # additional inputs
                                          pos=_pos,
                                          cross_attn=_cross_attn,
                                          cross_attn_mask=_cross_attn_mask)
            else:
                x = blk(x, attn_mask,
                        # additional inputs
                        pos=_pos,
                        cross_attn=_cross_attn,
                        cross_attn_mask=_cross_attn_mask) # t.s [B, h/4 * W/4 + 100, 48] ; ... ; [B, H/32 * W/32 + 100, 384]

        # reduce the number of patch tokens, but maintaining a fixed-scale det tokens
        # meanwhile, the channel dim increases by a factor of 2
        if self.downsample is not None:
            # divides both height and width by 2, increases last dimension (channels) by 4
            # then linearly project to prepare for next swin layer
            x_down = self.downsample(x, H, W) # t.s [B, H/8 * W/8 + 100, 96] ; ... ; [B, H/64 * W/64 + 100, 256]
            Wh, Ww = (H + 1) // 2, (W + 1) // 2 # new window height = ((H/4) + 1) // 2 ; ... ; ((H/32) + 1) // 2
                                                # new window width = ((W/4) + 1) // 2 ; ... ; ((W/32) + 1) // 2
            # returns:
            #   - x : output of swin block, before downsampling, t.s. [B, h/4 * W/4 + 100, 48] ; ... ; [B, H/32 * W/32 + 100, 384]
            #   - H : original passed in height 
            #   - W : original passed in width
            #   - x_down : downsampled features, patches have been merged for next layer, remember the hierarchical architecture, 
            #              t.s [B, H/8 * W/8 + 100, 96] ; ... ; [B, H/64 * W/64 + 100, 256]
            #   - Wh : new window height = ((H/4) + 1) // 2 ; ... ; ((H/32) + 1) // 2
            #   - Ww : new window width = ((W/4) + 1) // 2 ; ... ; ((W/32) + 1) // 2
            return x, H, W, x_down, Wh, Ww 
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Parameters:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        # patch_size = 4, in_chans=3, embed_dim=48 (only for Swin-nano), norm_layer=nn.LayerNorm
        super().__init__()
        patch_size = to_2tuple(patch_size) # (4, 4)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # input channels = 3
        # output channels = 48
        # kernel_size = 4
        # stride = 4
        # 
        # inp: [B, 3, H, W]; output: [B, 48, H/4, W/4]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # x = t.s [B, 3, H, W]
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0: # if width is not divisible by 4 (patch width)
            # pad the last dimension
            # pad to the left by 0, pad to the right by the remainder, so it's divisible by 4, 
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0: # if height is not divisble by 4 (patch height)
            # pad the last two dimensions,
            # last dimension (width) left and right padding is 0 so it is unchanged,
            # second last dimension (height) top padding is 0, bottom is padded by the remainder,
            # so padded height is divisible by 4
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        # for the sake of simplicity, H and W are still used to refer to the new padded heights and widths
        
        x = self.proj(x)  # t.s [B, 48, H/4, W/4]
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2) # t.s [B, H/4 * W/4, 48]
            x = self.norm(x) # t.s [B, H/4 * W/4, 48]
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)# t.s [B, 48, H/4, W/4]

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Parameters:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0., # FIXME: default is 0 but why?? experiment with this
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=[1, 2, 3], # NOTE: not used in the current version, please ignore.
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()
        # NOTE: the following instance variables were added by self.finetune_det() (see that to find values)
        # self.method, self.det_token_num, self.det_token, self.det_pos_embed, self.pos_dim, self.num_channels,
        # self.cross_indices, self.mask_divisor
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths) # 4
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape: # by default False so N/A
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            # fills input tensor with values drawn from a truncated normal dist. values must be in range (std ^ 2) from the mean
            # values outside this range are resized to fit into the range
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # NOTE: NEW: stochastic depth
        # since drop_path_rate = 0.0, dpr=[0.0] * sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # in range(4)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # modified by ViDT
                # modification: in vanilla swin, downsampling only done up to and excluding the last layer,
                #               meanwhile here, downsampling is performed up to and including the last layer
                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                last=None if (i_layer < self.num_layers-1) else True,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # [46, 96, 192, 384]
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        # NOTE: Not used in the current version -> please ignore. this error will be fixed later
        # we leave this lines to load the pre-trained model ...
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'det_pos_embed', 'det_token'}

    def finetune_det(self, method, det_token_num=100, pos_dim=256, cross_indices=[3]):
        """ A funtion to add neccessary (leanable) variables to Swin Transformer for object detection

            Parameters:
                method: vidt or vidt_wo_neck
                det_token_num: the number of object to detect, i.e., number of object queries
                pos_dim: the channel dimension of positional encodings for [DET] and [PATCH] tokens
                cross_indices: the indices where to use the [DET X PATCH] cross-attention
                    there are four possible stages in [0, 1, 2, 3]. 3 indicates Stage 4 in the ViDT paper.
        """
        # method = 'vidt', 
        # det_token_num=100, 
        # pos_dim=256, 
        # cross_indices=[3]

        # which method?
        self.method = method

        # how many object we detect?
        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.num_features[0])) # shape: [1, DET_NUM, 48]
        # fills input tensor with values drawn from a truncated normal dist. values must be in range (std ^ 2) from the mean
        # values outside this range are resized to fit into the range
        self.det_token = trunc_normal_(self.det_token, std=.02)

        # dim size of pos encoding
        self.pos_dim = pos_dim

        # learnable positional encoding for detection tokens
        det_pos_embed = torch.zeros(1, det_token_num, pos_dim) # shape: [1, 100, 256] 
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        self.det_pos_embed = torch.nn.Parameter(det_pos_embed)

        # info for detection
        # [96, 192, 384, 256 (appended below)]
        self.num_channels = [self.num_features[i+1] for i in range(len(self.num_features)-1)]
        if method == 'vidt':
            self.num_channels.append(self.pos_dim) # default: 256 (same to the default pos_dim)
        self.cross_indices = cross_indices
        # divisor to reduce the spatial size of the mask
        self.mask_divisor = 2 ** (len(self.layers) - len(self.cross_indices)) # 8

        # projection matrix for det pos encoding in each Swin layer (there are 4 blocks)
        for layer in self.layers:
            layer.det_token_num = det_token_num
            if layer.downsample is not None:
                layer.downsample.det_token_num = det_token_num
            for block in layer.blocks:
                block.det_token_num = det_token_num
                block.det_pos_linear = nn.Linear(pos_dim, block.dim)

        # neck-free model do not require downsamling at the last stage.
        if method == 'vidt_wo_neck':
            self.layers[-1].downsample = None

    def forward(self, x, mask):
        """ Forward function.

            Parameters:
                x: input rgb images
                mask: input padding masks [0: rgb values, 1: padded values]

            Returns:
                patch_outs: multi-scale [PATCH] tokens (four scales are used)
                    these tokens are the first input of the neck decoder
                det_tgt: final [DET] tokens obtained at the last stage
                    this tokens are the second input of the neck decoder
                det_pos: the learnable pos encoding for [DET] tokens.
                    these encodings are used to generate reference points in deformable attention
        """
        # x = t.s [B, 3, H, W], mask = t.s [B, H, W]

        # original input shape
        B, ori_H, ori_W = x.shape[0], x.shape[2], x.shape[3]

        # patch embedding
        # converts the images to patches using Conv2d
        x = self.patch_embed(x) # t.s [B, 48, H/4, W/4]

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2) # t.s [B, H/4 * W/4, 48]
        x = self.pos_drop(x) # dropout

        # expand det_token for all examples in the batch
        det_token = self.det_token.expand(B, -1, -1) # nn.Parameter(t.s [B, 100, 48])

        # det pos encoding -> will be projected in each block
        det_pos = self.det_pos_embed

        # prepare a mask for cross attention
        # "mask[None]" -> t.s [1,B,H,W]
        # size = (H/4 / 8, W/4 / 8) = (H/32, W/32)
        # downsamples the mask to [B, H/32, W/32]
        mask = F.interpolate(mask[None].float(),
                     size=(Wh // self.mask_divisor, Ww // self.mask_divisor)).to(torch.bool)[0]

        patch_outs = []
        for stage in range(self.num_layers): # in range(4)
            # stage = 0 ; 1 ; 2 ; 3
            layer = self.layers[stage]

            # whether to use cross-attention
            # cross attention only for last layer so only True when stage == 3
            cross_attn = True if stage in self.cross_indices else False

            # concat input
            # t.s [B, H/4 * W/4 + 100, 48] ; [B, H/8 * W/8 + 100, 96] ; [B, H/16 * W/16 + 100, 192] ; [B, H/32 * W/32 + 100, 384]
            x = torch.cat([x, det_token], dim=1) 

            # inference
            # returns:
            #   - x_out : output of swin block, before downsampling, 
            #             t.s. [B, h/4 * W/4 + 100, 48] ; [B, H/8 * W/8 + 100, 96] ; [B, H/16 * W/16 + 100, 192] ; [B, H/32 * W/32 + 100, 384]
            #   - H : original passed in height, H/4 ; H/8 ; H/16 ; H/32
            #   - W : original passed in width, W/4 ; W/8 ; W/16 ; W/32
            #   - x : downsampled features, patches have been merged for next layer, remember the hierarchical architecture, 
            #              t.s [B, H/8 * W/8 + 100, 96] ; [B, H/16 * W/16 + 100, 192] ; [B, H/32 * W/32 + 100, 384] ; [B, H/64 * W/64 + 100, 256]
            #   - Wh : new window height = ((H/4) + 1) // 2 ; ((H/8) + 1) // 2 ; ((H/16) + 1) // 2 ; ((H/32) + 1) // 2
            #   - Ww : new window width = ((W/4) + 1) // 2 ; ((W/8) + 1) // 2 ; ((W/16) + 1) // 2 ; ((W/32) + 1) // 2
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww,
                                           # additional input for VIDT
                                           input_mask=mask,
                                           det_pos=det_pos,
                                           cross_attn=cross_attn)

            # separate the [PATCH] and [DET] tokens
            # x t.s. [B, H/8 * W/8, 96] ; ... ; [B, H/64 * W/64, 256], 
            # det_token t.s [B, 100, 96] ; ... ; [b, 100, 256]
            x, det_token = x[:, :-self.det_token_num, :], x[:, -self.det_token_num:, :]

            # Aggregate intermediate outputs
            if stage > 0:
                # "x_out[:, :-self.det_token_num, :]" -> [B, H/8 * W/8, 96] ; [B, H/16 * W/16, 192] ; [B, H/32 * W/32, 384]
                # ".view()" -> [B, H/8, W/8, 96] ; [B, H/16, W/16, 192] ; [B, H/32, W/32, 384]
                # ".permute()" -> [B, 96, H/8, W/8] ; [B, 192, H/16, W/16] ; [B, 384, H/32, W/32]
                patch_out = x_out[:, :-self.det_token_num, :].view(B, H, W, -1).permute(0, 3, 1, 2)
                patch_outs.append(patch_out)

        # patch token reduced from last stage output
        # ".view()" -> [B, H/64, W/64, 256]
        # ".permute" -> [B, 256, H/64, W/64]
        patch_outs.append(x.view(B, Wh, Ww, -1).permute(0, 3, 1, 2))

        # det token
        # "x_out[:, -self.det_token_num:, :]" -> isolate the det part [B, 100, 384]
        # "permute" -> [B, 384, 100]
        det_tgt = x_out[:, -self.det_token_num:, :].permute(0, 2, 1)

        # det token pos encoding
        det_pos = det_pos.permute(0, 2, 1) # Parameter(t.s. [1, 256, 100])

        # returns:
        #   -patch_outs: intermediate [PATCH] token outputs from the second layer onwards, multi-scale feautures,
        #                list of size **4** (last two originates from the same layer 
        #                with the very last one undergoing further Linear projections).
        #                where each element is a tensor of shapes: 
        #                [B, 96, H/8, W/8] ; [B, 192, H/16, W/16] ; [B, 384, H/32, W/32] ; [B, 256, H/64, W/64]
        #   -det_tgt: final refined [DET] tokens, it is a tensor of shape: [B, 384, 100]
        #   -det_pos: final [DET] token learnable positional encoding, it is a Parameter, a tensor of shape: [1, 256, 100]
        return patch_outs, det_tgt, det_pos

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    # not working in the current version
    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def swin_nano(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=48, depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.0, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 384

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                    url="https://github.com/naver-ai/vidt/releases/download/v0.1-swin/swin_nano_patch4_window7_224.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')

        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')

    return model, 384


def swin_tiny(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=96, depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.2, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 768

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 768


def swin_small(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=96, depths=[2, 2, 18, 2],
                            num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.3, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 768

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
    return model, 768


def swin_base_win7(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=128, depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32], window_size=7, drop_path_rate=0.3, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024


def swin_large_win7(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=192, depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48], window_size=7, drop_path_rate=0.3, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024
