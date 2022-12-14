# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from util.misc import inverse_sigmoid
from ops.modules import MSDeformAttn

from timm.models.layers import DropPath


class DeformableTransformer(nn.Module):
    """ A Deformable Transformer for the neck in a detector

    NOTE: The transformer encoder is completely removed for ViDT
    Parameters:
        d_model: the channel dimension for attention [default=256]
        nhead: the number of heads [default=8]
        num_decoder_layers: the number of decoding layers [default=6]
        dim_feedforward: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        return_intermediate_dec: whether to return all the indermediate outputs [default=True]
        num_feature_levels: the number of scales for extracted features [default=4]
        dec_n_points: the number of reference points for deformable attention [default=4]
        NOTE: NEW params below
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
        token_label: whether to use the token label loss for training [default=False]. This is an additional trick
            proposed in  https://openreview.net/forum?id=LhbD74dsZFL (ICLR'22) for further improvement
    """

    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=True, num_feature_levels=4, dec_n_points=4,
                 drop_path=0., token_label=False):
        # d_model=256, 
        # nhead=8, 
        # num_decoder_layers=6, 
        # dim_feedforward=1024, 
        # dropout=0.1,
        # activation="relu", 
        # return_intermediate_dec=True, 
        # num_feature_levels=4, 
        # dec_n_points=4,
        # drop_path=0., 
        # token_label=False
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points,
                                                          drop_path=drop_path)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model)) # shape: [4, 256]
        self.token_label = token_label

        # (256, 2)
        self.reference_points = nn.Linear(d_model, 2)

        if self.token_label: # ViDT+, N/A
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            self.token_embed = nn.Linear(d_model, 91)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.token_embed.bias.data = torch.ones(91) * bias_value

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        # memory = concatenated [PATCH] tokens from the backbone, tensor of shape [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64), 256], 
        # memory_padding_mask = mask which indicates which values are padded (1) and which are not (0), 
        #                       tensor of shape [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64)], 
        # spatial_shapes = [H, W] for each scale [[H/8, W/8], [H/16, W/16], [H/32, W/32], [H/64, W/64]], tensor of shape [4, 2]
        
        N_, S_, C_ = memory.shape # N_ = B
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # gets the mask for level lvl
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1) # t.s. [B, H/8, W/8, 1] ; [B, H/16, W/16, 1] ; ...
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1) # get the number of non-padded values in the y-axis, i.e. valid height, t.s. [B]
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1) # get the number of non-padded values in the x-axis, i.e. valid width, t.s. [B]

            # "torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device)" -> [0, 1, 2, 3, ..., H_ - 1]
            # "torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device)" -> [0, 1, 2, 3, ..., W_ - 1]
            # grid_y is a grid of valid y-coordinates, it is a tensor of shape [H_, W_], it has form:
            #       [[0, 0, 0, ... , 0],
            #        [1, 1, 1, ... , 1],
            #           ...
            #        [H_ - 1, H_ - 1, ..., H_ - 1]]
            # grid_x is a grid of valid x-coordinates, it is a tensor of shape [H_, W_], it has form:
            #       [[0, 1, 2, ..., W_ - 1],
            #        [0, 1, 2, ..., W_ - 1],
            #           ...
            #        [0, 1, 2, ..., W_ - 1]]
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            
            # pair up the possible y's with the possible x's
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # t.s. [H_, W_, 2]
            # ".cat" -> [B, 2]
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2) # t.s [B, 1, 1, 2]
            # ".expand" -> [B, H_, W_, 2]
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale # [B, H_, W_, 2]
            # width and height
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl) # t.s. [B, H_, W_, 2] all values are the same
            # ".cat" -> [B, H_, W_, 4] => reminiscent of the bbox [cx, cy, w, h]
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4) # t.s. [B, H_ * W_, 4]
            proposals.append(proposal)
            _cur += (H_ * W_)
            
        output_proposals = torch.cat(proposals, 1) # t.s [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64), 4]
        # ensure that all proposed bbox values, cx, cy, w, h are between 0.01 and 0.99
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # reverse sigmoid (this was why all x's needed to be between 0 and 1)
        
        # in positions where mask = 1, i.e. padded, fill with "inf"
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # in positions where output proposals are invalid = 1 (after ~), fill with "inf"
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        ## in positions where mask = 1, i.e. padded, or output proposal is invalid, fill with 0s
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        # 
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        """
        Params:
            mask: 1 corresponding to padded values, 0 otherwise, tensor of shape [B, H, W]
        
        Returns:
            valid_ratio : for each image in the batch, it is the ratio of non-padded values to total H and W, tensor of shape [B, 2]
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, tgt, query_pos):
        """ The forward step of the decoder

        Parameters:
            srcs: [Patch] tokens
            masks: input padding mask
            tgt: [DET] tokens
            query_pos: [DET] token pos encodings

        Returns:
            hs: calibrated [DET] tokens
            init_reference_out: init reference points
            inter_references_out: intermediate reference points for box refinement
            enc_token_class_unflat: info. for token labeling
        """
        # srcs = list of size **4** where each element is a tensor of shape:
        #        [B, 256, H/8, W/8] ; [B, 256, H/16, W/16] ; [B, 256, H/32, W/32] ; [B, 256, H/64, W/64],
        # masks = [B, H/8, W/8] ; [B, H/16, W/16] ; [B, H/32, W/32] ; [B, H/64, W/64], 
        # tgt = [B, 100, 256],
        # query_pos = Parameter(t.s. [1, 100, 256])

        # prepare input for the Transformer decoder
        src_flatten = [] # [B, H/8 * W/8, 256] ; [B, H/16 * W/16, 256] ; [B, H/32 * W/32, 256] ; [B, H/64 * W/64, 256]
        mask_flatten = [] # [B, H/8 * W/8] ; [B, H/16 * W/16] ; [B, H/32 * W/32] ; [B, H/64 * W/64]
        spatial_shapes = [] # [H/8, W/8] ; [H/16, W/16] ; [H/32, W/32] ; [H/64, W/64]
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # [B, H/8 * W/8, 256] ; [B, H/16 * W/16, 256] ; [B, H/32 * W/32, 256] ; [B, H/64 * W/64, 256]
            src = src.flatten(2).transpose(1, 2)
            # [B, H/8 * W/8] ; [B, H/16 * W/16] ; [B, H/32 * W/32] ; [B, H/64 * W/64]
            mask = mask.flatten(1)
            src_flatten.append(src)
            mask_flatten.append(mask)
            
        # [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64), 256]
        src_flatten = torch.cat(src_flatten, 1)
        # [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64)]
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # t.s. [4, 2]
        # "spatial_shapes.new_zeros((1, ))" -> tensor([0])
        # "spatial_shapes.prod(1)" -> w*h for all different scales, i.e. areas of the different scales, t.s [4]
        # ".cumsum(0)[:-1]" -> [H/8*W/8, (H/8*W/8) + (H/16*W/16), (H/8*W/8) + (H/16*W/16) + (H/32*W/32)]
        # ".cat" -> [0, H/8*W/8, (H/8*W/8) + (H/16*W/16), (H/8*W/8) + (H/16*W/16) + (H/32*W/32)]
        # these are the indices to the beginning of the values for each of the levels, 
        # this is required to distinguish between the different levels due to the concatenation
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # t.s. [4]
    
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # t.s. [B, 4 2]

        memory = src_flatten
        bs, _, c = memory.shape
        tgt = tgt # [DET] tokens
        query_pos = query_pos.expand(bs, -1, -1) # [DET] token learnable positional encodings, t.s. [B, 100, 256]

        # prepare input for token label
        if self.token_label: # N/A
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_token_class_unflat = None
        if self.token_label: # N/A
            enc_token_class = self.token_embed(output_memory)
            enc_token_class_unflat = []
            for st, (h, w) in zip(level_start_index, spatial_shapes):
                enc_token_class_unflat.append(enc_token_class[:, st:st+h*w, :].view(bs, h, w, 91))

        # reference points for deformable attention
        # linearly project the learnable positional encodings for [DET] tokens from the backbone
        # to have dimension of 2, a pair of (x, y) reference points
        # scale it to be between 0-1
        reference_points = self.reference_points(query_pos).sigmoid() # t.s [B, 100, 2]
        init_reference_out = reference_points # query_pos -> reference point

        # decoder
        # returns:
        #   hs = tgt passed in + intermediate outputs from all decoder layers, t.s. [7, B, 100, 256]
        #   inter_references = reference_points passed in + reference points from all decoder layers, t.s. [7, B, 100, 4]
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_pos, mask_flatten)

        inter_references_out = inter_references
        
    
        # returns:
        #   hs = initial [DET] tokens from Swin backbone which is fed into the decoder + 
        #        intermediate outputs from all decoder layers, t.s. [7, B, 100, 256]
        #   init_reference_out = initial reference points, generated from linearly projecting 
        #                        [DET] token positional embeddings from the swin backbone, t.s. [B, 100, 2]
        #   inter_references_out = init_reference_out bbox refined + all reference points refined in each layer of the decoder,
        #                          t.s. [7, B, 100, 4]
        #   enc_token_class_unflat = None
        return hs, init_reference_out, inter_references_out, enc_token_class_unflat


class DeformableTransformerDecoderLayer(nn.Module):
    """ A decoder layer.

    Parameters:
        d_model: the channel dimension for attention [default=256]
        d_ffn: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        n_levels: the number of scales for extracted features [default=4]
        n_heads: the number of heads [default=8]
        n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, drop_path=0.):
        # d_model=256, 
        # d_ffn=1024,
        # dropout=0.1, 
        # activation="relu",
        # n_levels=4, 
        # n_heads=8, 
        # n_points=4, 
        # drop_path=0.
        super().__init__()

        # [DET x PATCH] deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # [DET x DET] self-attention
        # embedding dim = 256, num heads = 8, dropout = 0.1
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn for multi-heaed
        # in channel = 256, out channels = 1024
        self.linear1 = nn.Linear(d_model, d_ffn)
        # F.relu
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        # (1024, 256)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        # tgt = [B, 100, 256]
        # "linear1" -> [B, 100, 1024]
        # "linear2" -> [B, 100, 256]
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt)))) # t.s. [B, 100, 256]
        tgt = tgt + self.dropout4(tgt2) # residual connection 
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # tgt = [DET] tokens from backbone (first layer) / previous decoder layer output (second layer onwards), t.s. [B, 100, 256], 
        # query_pos = the positional encodings for the [DET] tokens, t.s. [B, 100, 256], 
        # reference_points = reference points between 0 and 1 which is wrt the actual non-padded values for all 4 scales for the image, t.s. [B, 100, 4, 4], 
        # src = [PATCH] tokens from the backbone, t.s. [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64), 256], 
        # src_spatial_shapes = the spatial shape, (H, W) of each multi-scale feature map t.s. [4,2], 
        # level_start_index = start index for every level of the multi-scale feature map, 
        #   used to distinguish between the features of each level from the concatenated src,
        #   tensor([0, H/8*W/8, (H/8*W/8) + (H/16*W/16), (H/8*W/8) + (H/16*W/16) + (H/32*W/32)]), 
        # src_padding_mask=input padding mask 1=padded, 0=non-padded, t.s. [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64)]
        
        
        ## [DET] self-attention
        # [DET] tokens + [DET] token positional encodings
        q = k = self.with_pos_embed(tgt, query_pos) # t.s. [B, 100, 256]
        # expected format for q is [L, B, E_q], where L is the target sequence length, B is the batch size, and E_q is the query embedding dim, t.s. [100, B, 256]
        # for k is [S, B, E_k], where S is the source sequence length, B is the batch size, and E_k is the key embedding dim, t.s. [100, B, 256]
        # for v is [S, B, E_v], where S is the source sequence length, B is the batch size, and E_v is the value embedding dim, t.s. [100,B,256]
        # output is a tuple of length 2 where the first element is a tensor of the attention outputs 
        #   of shape [L, B, E] where E is the embed_dim set when instantiating the nn.MultiheadAttention module
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1) # t.s. [B, 100, 256]
        
        # residual connection
        tgt = tgt + self.dropout2(tgt2) # t.s. [B, 100, 256]
        tgt = self.norm2(tgt)

        # Multi-scale deformable cross-attention in Eq. (1) in the ViDT paper
        # result of attention + positional encodings, t.s. [B, 100, 256]
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask) # t.s. [B, 100, 256]

        if self.drop_path is None: # here
            # residual connection
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # ffn
            tgt = self.forward_ffn(tgt) # t.s. [B, 100, 256]
        else:
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.drop_path(self.dropout4(tgt2))
            tgt = self.norm3(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    """ A Decoder consisting of multiple layers

    Parameters:
        decoder_layer: a deformable decoding layer
        num_layers: the number of layers
        return_intermediate: whether to return intermediate resutls
    """

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        #  decoder_layer = DeformableTransformerDecoderLayer object, 
        # num_layers = 6, 
        # return_intermediate=True
        super().__init__()
        # ModuleList with 6 deep copies of the decoder_layer
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement
        # this is set later as a ModuleList of 7 deep copies of MLP(hidden_dim, hidden_dim, 4, 3) = MLP(256, 256, 4, 3)
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """ The forwared step of the Deformable Decoder

        Parameters:
            tgt: [DET] tokens
            reference_poitns: reference points for deformable attention
            src: the [PATCH] tokens flattened into a 1-d sequence
            src_spatial_shapes: the spatial shape of each multi-scale feature map
            src_level_start_index: the start index to refer different scale inputs
            src_valid_ratios: the ratio of multi-scale feature maps
            query_pos: the pos encoding for [DET] tokens
            src_padding_mask: the input padding mask

        Returns:
            output: [DET] tokens calibrated (i.e., object embeddings)
            reference_points: A reference points

            If return_intermediate = True, output & reference_points are returned from all decoding layers
        """
        # tgt = [B, 100, 256], 
        # reference_points = [B, 100, 2] note: these values were passed through a sigmoid(),
        # src = [B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64), 256], 
        # src_spatial_shapes = [4,2], 
        # src_level_start_index = tensor([0, H/8*W/8, (H/8*W/8) + (H/16*W/16), (H/8*W/8) + (H/16*W/16) + (H/32*W/32)]), 
        # src_valid_ratios = [B, 4, 2], 
        # query_pos=[B, 100, 256], src_padding_mask=[B, (H/8*W/8) + (H/16*W/16) + (H/32*W/32) + (H/64*W/64)]

        output = tgt
        ## both are lists of size 7
        intermediate = [] # [B, 100, 256] (backbone [DET] token output); ... ; [B, 100, 256] (decoder output)
        intermediate_reference_points = [] # [B, 100, 4]

        # iterative bounding box refinement (handling the [DET] tokens produced from Swin with RAM)
        if self.bbox_embed is not None:
            tmp = self.bbox_embed[0](output) # t.s. [B, 100, 4]
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else: # here
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                # modify the (cx, cy) of the predicted bbox with the raw reference points (x,y)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid() # sigmoid() to scale everything to 0-1
            reference_points = new_reference_points.detach() # t.s. [B, 100, 4] note that now, reference_points.shape[-1] == 4 regardless

        if self.return_intermediate:
            intermediate.append(output)
            intermediate_reference_points.append(reference_points) 

        for lid, layer in enumerate(self.layers): # there are 6 deformable transformer decoder layers
            if reference_points.shape[-1] == 4:
                # "reference_points[:, :, None]"-> [B, 100, 1, 4]
                # ".cat()[:,None]" -> [B, 1, 4, 4] , where the last dimension consists of 2x repeated (valid_H, valid_W) pairs
                # reference_points for each image are now wrt the actual non-padded dimensions for all 4 scales
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # t.s [B, 100, 4, 4]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            # deformable operation
            # returns output t.s. [B, 100, 256]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                # self.bbox_embed is a list of size 7, not 6, so even it indexes with lid+1 = 6, it won't go out of bounds
                tmp = self.bbox_embed[lid+1](output) # t.s. [B, 100, 4]
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            # intermediate t.s. [7, B, 100, 256]
            # intermediate_reference_points, t.s. [7, B, 100, 4]
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        # return the output and reference points from the last decoder layer
        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.reduced_dim,
        nhead=args.nheads,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        token_label=args.token_label)


