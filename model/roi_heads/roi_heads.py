import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import heapq
import operator

from structures.instances import Instances
from structures.image_list import ImageList
from structures.box import Boxes,pairwise_iou
from layers.shape_spec import ShapeSpec


from ..backbone.resnet import BottleneckBlock, ResNet
from .fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference
from .fast_rcnn_mln import FastRCNNOutputLayers_MLN
from .box_head import build_box_head
from ..pooler import ROIPooler
from ..matcher import Matcher
from ..sampling import subsample_labels, subsample_labels_unknown
from model.rpn.utils import add_ground_truth_to_proposals
from layers.wrappers import cat
# from model.roi_heads.unkown_sample import acquisition_function

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    if cfg.MODEL.ROI_HEADS.NAME == 'Res5ROIHeads':
        return Res5ROIHeads(cfg,input_shape)
    else: 
        return StandardROIHeads(cfg,input_shape)

class ROIHeads(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.positive_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.proposal_matcher = Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            )
        self.proposal_append_gt = True
        self.auto_labeling_rpn = cfg.MODEL.RPN.AUTO_LABEL

    def _sample_proposals(self,matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
                     ,proposal: torch.Tensor = None,) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            matched_idxs : length N, best-matched gt index in [0,M) for each proposal
            matched_labels: length N, the matcher's label
        Returns:
            indices of sampled proposals 
            GT labels for sampled proposals
        '''
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
                )
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        # gt_classes_ss = gt_classes[sampled_idxs]
        # if self.auto_labeling:
        #     objectness_logits = proposal.objectness_logits
        #     matched_labels_ss = matched_labels[sampled_idxs]
        #     pred_objectness_score_ss = objectness_logits[sampled_idxs]
        #     if self.MDN:
        #         unct = (proposal.epis,proposal.alea)
        #     else:
        #         unct = None
        #     # 1) Remove FG objectness score. 2) Sort and select top k. 3) Build and apply mask.
        #     mask = torch.zeros((pred_objectness_score_ss.shape), dtype=torch.bool)
        #     pred_objectness_score_ss[matched_labels_ss != 0] = -1 # only background
        #     # sorted_indices = list(zip(
        #     #     *heapq.nlargest(2, enumerate(pred_objectness_score_ss), key=operator.itemgetter(1))))[0]
        #     # print(sorted_indices)
        #     sorted_indices = acquisition_function(pred_objectness_score_ss,unct,type=self.AF_TYPE)
        #     # print(sorted_indices)
        #     for index in sorted_indices:
        #         mask[index] = True
        #     gt_classes_ss[mask] = self.num_classes - 1
        # return sampled_idxs, gt_classes_ss
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
                self, proposals: List[Instances], targets: List[Instances]
            ) -> List[Instances]:
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
                , proposals_per_image
            )
            # print((gt_classes==20).sum())

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        # storage = get_event_storage()
        # storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        # storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        # log = {'roi_head/num_fg_samples' : np.mean(num_fg_samples),
        #         'roi_head/num_bg_samples': np.mean(num_bg_samples)}
        # wandb.log(log)
        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.
            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """
    def __init__(self, cfg, backbone_shape):
        super().__init__(cfg)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / backbone_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.res5, out_channels = self._build_res5_block(cfg)
        if cfg.MODEL.ROI_HEADS.USE_MLN:
            self.box_predictor = FastRCNNOutputLayers_MLN(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
                )
        else:
            print("standard")
            self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
                )

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels
        
    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        features= box_features.mean(dim=[2, 3])
        num_inst_per_image = [len(p) for p in proposals]
        features = features.split(num_inst_per_image, dim=0)
        
        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions,proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
    
    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        return instances
    
    # def extract_feature(self,
    #         features: Dict[str, torch.Tensor],
    #         proposals: List[Instances]):
    #     proposal_boxes = [x.proposal_boxes for x in proposals]
    #     box_features = self._shared_roi_transform(
    #         [features[f] for f in self.in_features], proposal_boxes
    #     )
    #     predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        
    #     features= box_features.mean(dim=[2, 3])
    #     num_inst_per_image = [len(p) for p in proposals]
    #     features = features.split(num_inst_per_image, dim=0)

    #     boxes = self.box_predictor.predict_boxes(predictions, proposals)
    #     scores = self.box_predictor.predict_probs(predictions, proposals)
    #     image_shapes = [x.image_size for x in proposals]

    #     _,feat,_ =  fast_rcnn_inference_f(
    #         boxes,
    #         scores,features,
    #         image_shapes,
    #         self.box_predictor.test_score_thresh,
    #         self.box_predictor.test_nms_thresh,
    #         self.box_predictor.test_topk_per_image,
    #     )
    #     res = cat(feat)
    #     return res
    #     # pred_instances, _ = self.box_predictor.inference(predictions, proposals)
    #     # pred_instances = self.forward_with_given_boxes(features, pred_instances)

class StandardROIHeads(ROIHeads):
    def __init__(self, cfg, backbone_shape):
        super().__init__(cfg)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / backbone_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        in_channels = [backbone_shape[f].channels for f in self.in_features]
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels= self.box_head.output_shape, height=1, width=1)
        )
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            return pred_instances, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def extract_feature(self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances]):
        num_inst_per_image = [len(p) for p in proposals]
        features = [features[f] for f in self.box_in_features]
        box_features = self.pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        return box_features.split(num_inst_per_image, dim=0)