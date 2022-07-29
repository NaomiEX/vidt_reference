# ------------------------------------------------------------------------
# DETR
# Copyright (c) 2020 Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally Modified by Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from methods.segmentation import (dice_loss, sigmoid_focal_loss)
import copy


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        # num_classes = 91, matcher = HungarianMatcher module, 
        # weight_dict = dict with loss names as keys and corresponding loss weightage as values, 
        #   ('loss_ce': 2, 'loss_bbox': 5, 'loss_giou' : 2, 'loss_iouaware' : 2) x 7 for final output + every intermediate layer, 
        #                                                                        x2 for encoder (though this model doesn't use encoder so it is unused)
        # losses = ['labels', 'boxes', 'cardinality', 'iouaware'], 
        # focal_alpha=0.25

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # outputs = dict containing:
        #   -pred_logits: logits for the predicted probabilities for the object category, t.s. [B, 100, 91]
        #   -pred_boxes: bbox prediction for the object, t.s. [B, 100, 4]
        #   -pred_ious: predicted iou for each of the bboxes, t.s. [B, 100, 1]
        #   -aux_outputs: list of size 6 where each element is a dict containing the key value pairs: 
        #                   "pred_logits", "pred_boxes", and "pred_ious" as described above 
        #                   for the [DET] tokens output from the swin backbone + each of the intermediate decoder layers, 
        # targets = list of size B with each element in the form:
        #   {'boxes' : torch.Size([N_o, 4]),
        #   'labels' : torch.Size([N_o]),
        #   'image_id' : torch.Size([1]),
        #   'xyxy_boxes' : torch.Size([N_o, 4]),
        #   'area' : torch.Size([N_o]),
        #   'iscrowd' : torch.Size([N_o]),
        #   'orig_size' : torch.Size([2]),
        #   'size' : torch.Size([2])}, 
        # indices = list of size B where each element is a tuple consisting of 2 elements:
        #       tensor of shape [N_o_i] which contains row indices,
        #       and a tensor of shape [N_o_i] which contains the matching col indices, 
        # num_boxes = avg. num boxes per device, log=True

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # t.s. [B, 100, 91]

        # idx is a tuple (batch_idx, src_idx),
        # batch_idx stores the batch of the src in src_idx in the corresponding position
        # for ex. [0,0,1,1,1, ...], it has t.s. [N_o_total]
        # src_idx stores the concatenated chosen row indices, these row indices index the predictions (100)
        # t.s. [N_o_total]
        idx = self._get_src_permutation_idx(indices)
        # col indices index the ground-truth labels to be matched up with
        # so here it concatenates the chosen labels in the order specified by the Hungarian Matching algo.
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # t.s. [N_o_total]
        # fill up a tensor of shape [B, 100] with the value 91
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # since idx is (batch_idx, src_idx), this assignment replaces the values in the chosen predictions' positions
        # with the target classes
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # t.s. [B, 100, 92]
        # replace the 0s in the indices indicated by target_classes by 1 in the last dimension (92)
        # why?
        #   because in target classes the default value, i.e. unchosen matchings have value 91, 
        #   since there are in total 91 object classes, the possible labels range from 0 to *90*, so no valid object would have label 91.
        #   target_classes_onehot's last dimension is of size 92 so this does not cause error
        #   Therefore, only for unchosen predictions, the 92-th value in the vector will be 1
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # unchosen predictions now have a vector of 91 0s
        target_classes_onehot = target_classes_onehot[:,:,:-1] # discard the last column, t.s. [B, 100, 91]
        # sigmoid_focal_loss returns the average mean focal loss across all object classes per ground truth bbox
        # multiplied by the number of predictions
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # outputs = dict containing:
        #   -pred_logits: logits for the predicted probabilities for the object category, t.s. [B, 100, 91]
        #   -pred_boxes: bbox prediction for the object, t.s. [B, 100, 4]
        #   -pred_ious: predicted iou for each of the bboxes, t.s. [B, 100, 1]
        #   -aux_outputs: list of size 6 where each element is a dict containing the key value pairs: 
        #                   "pred_logits", "pred_boxes", and "pred_ious" as described above 
        #                   for the [DET] tokens output from the swin backbone + each of the intermediate decoder layers, 
        # targets = list of size B with each element in the form:
        #   {'boxes' : torch.Size([N_o, 4]),
        #   'labels' : torch.Size([N_o]),
        #   'image_id' : torch.Size([1]),
        #   'xyxy_boxes' : torch.Size([N_o, 4]),
        #   'area' : torch.Size([N_o]),
        #   'iscrowd' : torch.Size([N_o]),
        #   'orig_size' : torch.Size([2]),
        #   'size' : torch.Size([2])}, 

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        # tensor containing the number of objects in each image
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # t.s. [B]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # "pred_logits.argmax(-1)" -> gets the indices of the maximum value of the last dimension (91) # t.s. [B, 100]
        # sum up the number of predictions for which the index with the highest probability is not 90 
        # (i.e. the 91-th or no-object class), t.s. [B]
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # get the l1 loss between the number of object predictions excluding "no-object"s against the ground truth number of objects
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # outputs = dict containing:
        #   -pred_logits: logits for the predicted probabilities for the object category, t.s. [B, 100, 91]
        #   -pred_boxes: bbox prediction for the object, t.s. [B, 100, 4]
        #   -pred_ious: predicted iou for each of the bboxes, t.s. [B, 100, 1]
        #   -aux_outputs: list of size 6 where each element is a dict containing the key value pairs: 
        #                   "pred_logits", "pred_boxes", and "pred_ious" as described above 
        #                   for the [DET] tokens output from the swin backbone + each of the intermediate decoder layers, 
        # targets = list of size B with each element in the form:
        #   {'boxes' : torch.Size([N_o, 4]),
        #   'labels' : torch.Size([N_o]),
        #   'image_id' : torch.Size([1]),
        #   'xyxy_boxes' : torch.Size([N_o, 4]),
        #   'area' : torch.Size([N_o]),
        #   'iscrowd' : torch.Size([N_o]),
        #   'orig_size' : torch.Size([2]),
        #   'size' : torch.Size([2])}, 
        # indices = list of size B where each element is a tuple consisting of 2 elements:
        #       tensor of shape [N_o_i] which contains row indices,
        #       and a tensor of shape [N_o_i] which contains the matching col indices, 
        # num_boxes = avg. num boxes per device,

        assert 'pred_boxes' in outputs
        # returns a tuple of (batch_idx, src_idx) both of t.s. [N_o_total]
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx] # t.s. [N_o_total, 4]
        # col indexes ground-truth bboxes
        # get and concatenate the bboxes from target for each image
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # t.s. [N_o_total, 4]

        # get the L1 distance between the predicted and the actual bboxes as a single value
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        # get the avg. loss per number of target boxes
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # get the giou for every predicted bbox against every target bbox,
        # ofc we only care about the diagonals because that is the one the prediction is matched to
        # therefore only extract the values from the main diagonal, t.s. [N_o_total]
        # "1-" -> done because if the predicted bbox does not overlap much with the target, iou is low, 
        # and we want loss to be high, so we do 1-iou
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        # get the average giou loss per target bbox
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_iouaware(self, outputs, targets, indices, num_boxes):
        # outputs = dict containing:
        #   -pred_logits: logits for the predicted probabilities for the object category, t.s. [B, 100, 91]
        #   -pred_boxes: bbox prediction for the object, t.s. [B, 100, 4]
        #   -pred_ious: predicted iou for each of the bboxes, t.s. [B, 100, 1]
        #   -aux_outputs: list of size 6 where each element is a dict containing the key value pairs: 
        #                   "pred_logits", "pred_boxes", and "pred_ious" as described above 
        #                   for the [DET] tokens output from the swin backbone + each of the intermediate decoder layers, 
        # targets = list of size B with each element in the form:
        #   {'boxes' : torch.Size([N_o, 4]),
        #   'labels' : torch.Size([N_o]),
        #   'image_id' : torch.Size([1]),
        #   'xyxy_boxes' : torch.Size([N_o, 4]),
        #   'area' : torch.Size([N_o]),
        #   'iscrowd' : torch.Size([N_o]),
        #   'orig_size' : torch.Size([2]),
        #   'size' : torch.Size([2])}, 
        # indices = list of size B where each element is a tuple consisting of 2 elements:
        #       tensor of shape [N_o_i] which contains row indices,
        #       and a tensor of shape [N_o_i] which contains the matching col indices, 
        # num_boxes = avg. num boxes per device,
        assert 'pred_ious' in outputs
        
        # idx is a tuple (batch_idx, src_idx)
        # where both are tensors of shape [N_o_total]
        idx = self._get_src_permutation_idx(indices)
        src_ious = outputs['pred_ious'][idx]  # gets the ious for the chosen predictions, t.s. [N_o_total, 1]
        src_ious = src_ious.squeeze(1) # t.s. [N_o_total]
        src_boxes = outputs['pred_boxes'][idx] # get the predicted boxes for the chosen predictions, t.s. [N_o_total, 4]
        # get the ground-truth bboxes
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # t.s. [N_o_total, 4]

        
        # compute the box iou (NOT Giou) between the predicted and actual
        # returns a tuple of 2 elements: iou, union (we are only interested in iou so just take the first elem)
        # again it returns the IoU between every predicted bbox and every gt bbox
        # so only take the diagonals
        iou = torch.diag(box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))[0]) # t.s. [N_o_total]

        losses = {}
        loss_iouaware = F.binary_cross_entropy_with_logits(src_ious, iou, reduction='none')
        losses['loss_iouaware'] = loss_iouaware.sum() / num_boxes # get the average IoU loss per ground truth bbox
        return losses

    def loss_tokens(self, outputs, targets, num_boxes):
        enc_token_class_unflat = outputs['pred_logits']

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()

        bs, n, h, w = target_masks.shape
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=target_masks.device)
        for j in range(n):
            target_masks[:, j] &= target_masks[:, j] ^ mask
            mask |= target_masks[:, j]
        target_classes_pad = torch.stack([F.pad(t['labels'], (0, n - len(t['labels']))) for t in targets])
        final_mask = torch.sum(target_masks * target_classes_pad[:, :, None, None], dim=1)  # (bs, h, w)
        final_mask_onehot = torch.zeros((bs, h, w, self.num_classes), dtype=torch.float32, device=target_masks.device)
        final_mask_onehot.scatter_(-1, final_mask.unsqueeze(-1), 1)  # (bs, h, w, 91)

        final_mask_onehot[..., 0] = 1 - final_mask_onehot[..., 0]  # change index 0 from background to foreground

        loss_token_focal = 0
        loss_token_dice = 0
        for i, enc_token_class in enumerate(enc_token_class_unflat):
            _, h, w, _ = enc_token_class.shape

            final_mask_soft = F.adaptive_avg_pool2d(final_mask_onehot.permute(0, 3, 1, 2), (h,w)).permute(0, 2, 3, 1)

            enc_token_class = enc_token_class.flatten(1, 2)
            final_mask_soft = final_mask_soft.flatten(1, 2)
            loss_token_focal += sigmoid_focal_loss(enc_token_class, final_mask_soft, num_boxes)
            loss_token_dice += dice_loss(enc_token_class, final_mask_soft, num_boxes)

        losses = {
            'loss_token_focal': loss_token_focal,
            'loss_token_dice': loss_token_dice,
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # indices = list of size B where each element is a tuple consisting of 2 elements:
        #       tensor of shape [N_o_i] which contains row indices,
        #       and a tensor of shape [N_o_i] which contains the matching col indices, 
        
        # permute predictions following indices
        
        # copy batch number N_o_i times for each image in the batch, for ex. if the first image has 2 objects
        # and the second image has 4, [0, 0, 1, 1, 1, 1, ...]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices]) # get only the row indices, the rows index the predictions
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Maps loss names to loss functions and execute it with the given parameters and kwargs

        Args:
            loss (str): the name of the loss
            outputs (dict): outputs from the model, 
            targets (List[dict]): the ground-truth values, a list of size B containing ground-truth organized as dicts
            indices (List(tuple(Tensor))): indices for matched pairs from Hungarian Matching, list of size B where 
                each element is a tuple of 2 tensors - row and col indices
            num_boxes (torch.float): avg. number of boxes per device

        Returns:
            loss: _description_
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality, 
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'iouaware': self.loss_iouaware,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, distil_tokens=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs = dict containing:
        #   -pred_logits: logits for the predicted probabilities for the object category, t.s. [B, 100, 91]
        #   -pred_boxes: bbox prediction for the object, t.s. [B, 100, 4]
        #   -pred_ious: predicted iou for each of the bboxes, t.s. [B, 100, 1]
        #   -aux_outputs: list of size 6 where each element is a dict containing the key value pairs: 
        #                   "pred_logits", "pred_boxes", and "pred_ious" as described above 
        #                   for the [DET] tokens output from the swin backbone + each of the intermediate decoder layers, 
        # targets = list of size B with each element in the form:
        #   {'boxes' : torch.Size([N_o, 4]),
        #   'labels' : torch.Size([N_o]),
        #   'image_id' : torch.Size([1]),
        #   'xyxy_boxes' : torch.Size([N_o, 4]),
        #   'area' : torch.Size([N_o]),
        #   'iscrowd' : torch.Size([N_o]),
        #   'orig_size' : torch.Size([2]),
        #   'size' : torch.Size([2])} , 
        # distil_tokens=None
        
        # extract non-aux outputs
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # returns:
        #   list of size B where each element is a tuple consisting of 2 elements:
        #       tensor of shape [N_o_i] which contains row indices,
        #       and a tensor of shape [N_o_i] which contains the matching col indices
        indices = self.matcher(outputs_without_aux, targets)
        _indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) # get the total number of boxes across all images in the batch
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes) # get the total across all devices
            
        # avg num of boxes per device
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            # loss = "labels"
            #   
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
            
        # losses:
        #   'loss_labels' : get the avg. mean focal loss across all object classes per ground truth bbox
        #   'loss_cardinality' : the l1 loss between the number of non-trivial (no-object) predictions and actual number of gt bboxes
        #   'loss_bbox' : get the avg. l1 bbox loss per ground truth bbox
        #   'loss_iouaware' : get the average IoU loss per ground truth bbox

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}

                    losses.update(l_dict)
        # losses:
        #   'loss_labels' : get the avg. mean focal loss across all object classes per ground truth bbox
        #   'loss_cardinality' : the l1 loss between the number of non-trivial (no-object) predictions and actual number of gt bboxes
        #   'loss_bbox' : get the avg. l1 bbox loss per ground truth bbox
        #   'loss_iouaware' : get the average IoU loss per ground truth bbox
        #   'loss_labels_0', 'loss_cardinality_0', 'loss_bbox_0', 'loss_iouaware_0', -> repeat the losses as described above for the [DET] tokens output from the swin backbone
        #       ...,
        #   'loss_labels_5', 'loss_cardinality_5', 'loss_bbox_5', 'loss_iouaware_5' -> repeat the losses as described above for the 6th and last intermediate decoder layer

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'enc_tokens' in outputs:
            l_dict = self.loss_tokens(outputs['enc_tokens'], targets, num_boxes)
            losses.update(l_dict)

        # distil. loss
        if distil_tokens is not None:
            patches, teacher_patches = distil_tokens[0]['patch_token'], distil_tokens[1]['patch_token']
            body_det, teacher_body_det = distil_tokens[0]['body_det_token'], distil_tokens[1]['body_det_token']
            neck_det, teacher_neck_det = distil_tokens[0]['neck_det_token'], distil_tokens[1]['neck_det_token']

            distil_loss = 0.0
            for patch, teacher_patch in zip(patches, teacher_patches):
                b, c, w, h = patch.shape
                patch = patch.permute(0, 2, 3, 1).contiguous().view(b*w*h, c)
                teacher_patch = teacher_patch.permute(0, 2, 3, 1).contiguous().view(b*w*h, c).detach()
                distil_loss += torch.mean(torch.sqrt(torch.sum(torch.pow(patch - teacher_patch, 2), dim=-1)))

            b, d, c = body_det.shape
            body_det = body_det.contiguous().view(b*d, c)
            teacher_body_det = teacher_body_det.contiguous().view(b*d, c).detach()
            distil_loss += torch.mean(torch.sqrt(torch.sum(torch.pow(body_det - teacher_body_det, 2), dim=-1)))

            l, b, d, c = neck_det.shape
            neck_det = neck_det.contiguous().view(l*b*d, c)
            teacher_neck_det = teacher_neck_det.contiguous().view(l*b*d, c).detach()
            distil_loss += (torch.mean(torch.sqrt(torch.sum(torch.pow(neck_det - teacher_neck_det, 2), dim=-1))) * l)

            l_dict = {'loss_distil': torch.sqrt(distil_loss)}
            losses.update(l_dict)

        return losses



