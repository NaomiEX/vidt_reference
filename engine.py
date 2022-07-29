# ------------------------------------------------------------------------
# DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    n_iter_to_acc: int = 1, print_freq: int = 100):
    """
    Training one epoch

    Parameters:
        model: a target model
        criterion: a critetrion module to compute training (or val, test) loss
        data_loader: a training data laoder to use
        optimizer: an optimizer to use
        epoch: the current epoch number
        max_norm: a max norm for gradient clipping (default=0)
        n_iter_to_acc: the step size for gradient accumulation (default=1)
        print_freq: the step size to print training logs (default=100)

    Return:
        dict: a log dictionary with keys (log type) and values (log value)
    """
    # model: torch.nn.Module = Detector module, 
    # criterion: torch.nn.Module = SetCriterion module,
    # data_loader: Iterable = DataLoader, 
    # optimizer: torch.optim.Optimizer = AdamW,
    # device: torch.device, 
    # epoch: int, 
    # max_norm: float = 0.1,
    # n_iter_to_acc: int = 1,
    # print_freq: int = 500

    model.train()
    criterion.train()

    # register log types
    metric_logger = utils.MetricLogger(delimiter=", ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = print_freq

    batch_idx = 0
    # iterate one epoch
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # samples = NestedTensor object with two attributes, 
        #   tensors = torch.Size([B, 3, H, W]), mask: torch.Size([B, H, W])
        samples = samples.to(device)
        # targets = list of size B with each element in the form:
        #   {'boxes' : torch.Size([N_o, 4]),
        #   'labels' : torch.Size([N_o]),
        #   'image_id' : torch.Size([1]),
        #   'xyxy_boxes' : torch.Size([N_o, 4]),
        #   'area' : torch.Size([N_o]),
        #   'iscrowd' : torch.Size([N_o]),
        #   'orig_size' : torch.Size([2]),
        #   'size' : torch.Size([2])} 
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # inference
        # dict containing:
        #   -pred_logits: logits for the predicted probabilities for the object category, t.s. [B, 100, 91]
        #   -pred_boxes: bbox prediction for the object, t.s. [B, 100, 4]
        #   -pred_ious: predicted iou for each of the bboxes, t.s. [B, 100, 1]
        #   -aux_outputs: list of size 6 where each element is a dict containing the key value pairs: 
        #                   "pred_logits", "pred_boxes", and "pred_ious" as described above 
        #                   for the [DET] tokens output from the swin backbone + each of the intermediate decoder layers
        outputs = model(samples)

        # compute loss
        # loss_dict:
        #   'loss_labels' : get the avg. mean focal loss across all object classes per ground truth bbox
        #   'loss_cardinality' : the l1 loss between the number of non-trivial (no-object) predictions and actual number of gt bboxes
        #   'loss_bbox' : get the avg. l1 bbox loss per ground truth bbox
        #   'loss_iouaware' : get the average IoU loss per ground truth bbox
        #   'loss_labels_0', 'loss_cardinality_0', 'loss_bbox_0', 'loss_iouaware_0', -> repeat the losses as described above for the [DET] tokens output from the swin backbone
        #       ...,
        #   'loss_labels_5', 'loss_cardinality_5', 'loss_bbox_5', 'loss_iouaware_5' -> repeat the losses as described above for the 6th and last intermediate decoder layer
        loss_dict = criterion(outputs, targets)
        
        # get the dictionary containing the weightage for each type of loss
        #   ('loss_ce': 2, 'loss_bbox': 5, 'loss_giou' : 2, 'loss_iouaware' : 2) x 6 for every intermediate layer, 
        weight_dict = criterion.weight_dict
        # get the total weighted summed losses
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # backprop.
        losses /= float(n_iter_to_acc)
        losses.backward()
        if (batch_idx + 1) % n_iter_to_acc == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()

        # save logs per iteration
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        batch_idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_with_teacher(model: torch.nn.Module, teacher_model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    n_iter_to_acc: int = 1, print_freq: int = 100):
    """
    Training one epoch

    Parameters:
        model: a target model
        teacher_model: a teacher model for distillation
        criterion: a critetrion module to compute training (or val, test) loss
        data_loader: a training data laoder to use
        optimizer: an optimizer to use
        epoch: the current epoch number
        max_norm: a max norm for gradient clipping (default=0)
        n_iter_to_acc: the step size for gradient accumulation (default=1)
        print_freq: the step size to print training logs (default=100)

    Return:
        dict: a log dictionary with keys (log type) and values (log value)
    """

    model.train()
    teacher_model.eval()
    criterion.train()

    # register log types
    metric_logger = utils.MetricLogger(delimiter=", ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = print_freq

    batch_idx = 0
    # iterate one epoch
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # inference
        outputs = model(samples)
        teacher_outputs = teacher_model(samples)

        # collect distillation token for matching loss
        distil_tokens = (outputs['distil_tokens'], teacher_outputs['distil_tokens'])

        # compute loss
        loss_dict = criterion(outputs, targets, distil_tokens=distil_tokens)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # backprop.
        losses.backward()
        if (batch_idx + 1) % n_iter_to_acc == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()

        # save logs per iteration
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        batch_idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device):
    """
    Training one epoch

    Parameters:
        model: a target model
        criterion: a critetrion module to compute training (or val, test) loss
        postprocessors: a postprocessor to compute AP
        data_loader: an eval data laoder to use
        base_ds: a base dataset class
        device: the device to use (GPU or CPU)

    Return:
        dict: a log dictionary with keys (log type) and values (log value)
    """

    model.eval()
    criterion.eval()

    # register log types
    metric_logger = utils.MetricLogger(delimiter=", ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # return eval. metrics
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # iterate for all eval. examples
    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # inference
        outputs = model(samples)

        # loss compute
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # compute AP, etc
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator
