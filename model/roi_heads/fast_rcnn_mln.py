from typing import Dict, List, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
import wandb

from layers.shape_spec import ShapeSpec
from layers.nms import batched_nms
from layers.wrappers import cat,nonzero_tuple

from model.backbone.mdn import MixtureOfLogits
from model.roi_heads.mln import mln_gather,mace_loss, mln_uncertainties
from model.roi_heads.unct_inference import filter_unct
from model.box_regression import Box2BoxTransform, _dense_box_regression_loss
from structures.box import Boxes
from structures.instances import Instances


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor], epis_unct: List[torch.Tensor],
    alea_unct: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int, path: str = None
):
    """
    Call `fast_rcnn_inference_single_image` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, epis_per_image, alea_per_image,image_shape, score_thresh, 
                nms_thresh, topk_per_image , path
        )
        for scores_per_image, epis_per_image, alea_per_image, boxes_per_image, image_shape in zip(scores, epis_unct, alea_unct,boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def _log_classification_stats(pred_logits, gt_classes, log=True , auto_labeling=True):
    """
    Log the classification metrics to EventStorage.
    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1
    if auto_labeling:
        fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind-1)
        unk_class_ind = bg_class_ind -1
        unk_inds = gt_classes == unk_class_ind
        num_unk = unk_inds.nonzero().numel()
        unk_gt_classes = gt_classes[unk_inds]
        unk_pred_classes = pred_classes[unk_inds]
        unk_num_accurate = (unk_pred_classes == unk_gt_classes).nonzero().numel()
        if num_unk == 0:
            num_unk += 1
    else:
        fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    # storage = get_event_storage()
    string = {'clss accuracy': num_accurate/num_instances}
    if log:
        wandb.log(string)
    # storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        if auto_labeling:
            string = {'fg_cls_accuracy' : fg_num_accurate / num_fg,
                'false_negative' :  num_false_negative / num_fg,
                'unk_cls_accuracy':unk_num_accurate/num_unk}
        else:
            string = {'fg_cls_accuracy' : fg_num_accurate / num_fg,
                'false_negative' :  num_false_negative / num_fg}
        if log:
            wandb.log(string)
        # storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        # storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


def fast_rcnn_inference_single_image(
    boxes,
    scores, epis, alea,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int, path: str = None
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """

    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        alea = alea[valid_mask]
        epis = epis[valid_mask]
    # print(boxes.shape,alea.shape,epis.shape)
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    epis = epis[filter_inds[:, 0]]
    alea = alea[filter_inds[:, 0]]
    
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    epis , alea = epis[keep], alea[keep]
    unct = epis + alea
    if path is not None:
        filter_inds[:, 1] = filter_unct(filter_inds[:, 1],unct,path,unct.device)
    
    pred_classes = filter_inds[:, 1]
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = pred_classes
    result.epis = epis
    result.alea = alea
    return result, filter_inds[:, 0]


class FastRCNNOutputLayers_MLN(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. proposal-to-detection box regression deltas
    2. classification scores
    """
    def __init__(
        self,cfg,
        input_shape: ShapeSpec):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.from_config(cfg,input_shape)

        self.cls_score.init_parameters()
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def from_config(self,cfg,input_shape):
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.cls_score = MixtureOfLogits(input_size, self.num_classes + 1)
        num_bbox_reg_classes = 1 if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG else self.num_classes
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.box_reg_loss_type = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE
        loss_weight = {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT}
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.log = cfg.log
        self.auto_labeling = cfg.MODEL.ROI_HEADS.AUTO_LABEL
        self.rpn_auto_labeling = cfg.MODEL.RPN.AUTO_LABEL

        RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
        ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
        MODEL_NAME = RPN_NAME + ROI_NAME
        if cfg.MODEL.ROI_HEADS.UNCT:
            self.path = '{}/ckpt/{}/{}_{}.json'.format(cfg.PATH,cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME)
        else:
            self.path = None
    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.
        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.
            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        gather_scores = mln_gather(scores)
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(gather_scores, gt_classes,self.log,self.auto_labeling+self.rpn_auto_labeling)
        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        targets_cls = torch.eye(self.num_classes+1)[gt_classes].to('cuda')
        losses = {
            "loss_cls": mace_loss(scores, targets_cls, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores , epis_unct, alea_unct = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,epis_unct,alea_unct,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image, path=self.path
        )
    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.
        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.
        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        gather_scores = mln_gather(scores)
        uncertainty = mln_uncertainties(scores['pi'],scores['mu'])
        epis = uncertainty['epis']
        alea = uncertainty['alea']
        num_inst_per_image = [len(p) for p in proposals]
        # probs = F.softmax(gather_scores, dim=-1)
        return gather_scores.split(num_inst_per_image, dim=0), epis.split(num_inst_per_image, dim=0),alea.split(num_inst_per_image, dim=0)