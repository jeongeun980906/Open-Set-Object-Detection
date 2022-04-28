from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
import wandb

from layers.wrappers import Conv2d, cat
from structures.box import Boxes,pairwise_iou
from structures.image_list import ImageList
from structures.instances import Instances

from tools_det.memory import retry_if_cuda_oom
from model.rpn.utils import find_top_rpn_proposals, find_top_rpn_proposals_uncertainty
from ..anchor_generator import build_anchor_generator
from ..matcher import Matcher,Matcher2
from ..sampling import subsample_labels
from ..box_regression import Box2BoxTransform,_dense_box_regression_loss
from fvcore.nn import smooth_l1_loss
from .mdn_loss import mdn_loss,find_logit,mdn_uncertainties
from model.backbone.mdn import MixtureHead_CNN

from model.ssl_score.dino_score import autolabel_dino
from model.ssl_score.preprocess import open_candidate
from model.ssl_score.append_gt import append_gt

class RPNHead_MDN(nn.Module):
    def __init__(self,in_channels,num_anchors,box_dim = 4,conv_dims = (-1,)):
        super().__init__()
        cur_channels = in_channels
        if len(conv_dims)==1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            self.conv = self._rpn_conv(cur_channels,out_channels)
            cur_channels = out_channels
        else:
            ''''
            FPN network
            '''
            raise NotImplementedError
        self.objectness = MixtureHead_CNN(cur_channels,num_anchors)
        self.anchor_deltas = nn.Conv2d(cur_channels,num_anchors*box_dim,kernel_size=1,stride=1)

        # Weight Init
        for layers in self.modules():
            if isinstance(layers,nn.Conv2d):
                nn.init.normal(layers.weight,std=0.01)
                nn.init.normal(layers.bias,0)

    def _rpn_conv(self,in_channels,out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )
    def forward(self,features: List[torch.Tensor]):
        '''
        Args:
            features list[Tensor] 
        Returns:
            predicted objectness list[Dict[Tensor]] 
            predicted anchor box deltas list[Tensor]
        '''
        pred_objectness = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness.append(self.objectness(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness,pred_anchor_deltas


class RPN_MDN(nn.Module):
    def __init__(self,in_features:List[str], head:nn.Module
                    ,anchor_generator:nn.Module, anchor_matcher : Matcher,
                    box2box_transform: Box2BoxTransform,
                    batch_size_per_image: int,
                    positive_fraction: float,
                    pre_nms_topk: Tuple[float, float],
                    post_nms_topk: Tuple[float, float],
                    nms_thresh: float = 0.7,
                    min_box_size: float = 0.0,
                    anchor_boundary_thresh: float = -1.0,
                    loss_weight: Union[float, Dict[str, float]] = 1.0,
                    box_reg_loss_type: str = "smooth_l1",
                    smooth_l1_beta: float = 0.0 ,auto_labeling: bool = False,
                    auto_label_type: str = 'mul',
                    log:bool=True):

        super().__init__()
        self.in_features = in_features # FPN ['p2','p3'...] resnet ['res4']
        self.anchor_generator = anchor_generator
        self.head = head
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.log = log
        self.auto_labeling  = auto_labeling
        if auto_labeling:
            self.auto_labeling_type = auto_label_type
            self.MODEL = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            self.candidate_set = open_candidate()
            self.auto_label_matcher = Matcher(
                [0.3, 0.7],[0, -1, 1], allow_low_quality_matches=False)

    def forward(self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[Instances]] = None):
        '''
        Args:
            images
            features
            gt_instances
        Returns:
            Proposals
            Loss
        '''
        # extract feature
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.head(features)

        # Transform
        pred_objectness_logits = {'pi':[
            # (N, K, A, Hi, Wi) -> (N, K, Hi, Wi, A) -> (N, K, Hi*Wi*A)
            score['pi'].permute(0, 1, 3, 4, 2).flatten(2)
            for score in pred_objectness_logits],
            'mu':[
                # (N, K, A, Hi, Wi) -> (N, K, Hi, Wi, A) -> (N, K, Hi*Wi*A)
                score['mu'].permute(0, 1, 3, 4, 2).flatten(2)
                for score in pred_objectness_logits],
            'sigma':[
                # (N, K, A, Hi, Wi) -> (N, K, Hi, Wi, A) -> (N, K, Hi*Wi*A)
                score['sigma'].permute(0, 1, 3, 4, 2).flatten(2)
                for score in pred_objectness_logits]

        }
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        
        proposals = self.predict_proposals_uncertainty(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        if self.auto_labeling:
            with torch.no_grad():
                label = autolabel_dino(images,proposals,gt_instances,self.auto_label_matcher
                        ,self.MODEL,self.candidate_set,score_type = self.auto_labeling_type)
                gt_instances = append_gt(label,gt_instances)
        if self.training:
            gt_labels,gt_targets,  gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels,gt_targets, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}

        if self.auto_labeling:
            return proposals,losses,gt_instances
        else:
            return proposals, losses

    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        '''
        Args:
            anchors
            gt_instances
        Returns:
            matched gt labels
            matched gt boxes
        '''
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        gt_targets = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i, gt_targets_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix) # [anchor h/16 w/16]
            gt_targets_i = gt_targets_i.to(device=gt_targets_i.device)
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)
            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            gt_targets.append(gt_targets_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels,gt_targets, matched_gt_boxes

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.
        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def losses(self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor], gt_targets: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor]):

        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        gt_targets = torch.stack(gt_targets)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        log = {'num_pos_anchors': num_pos_anchors,
                'num_neg_anchors': num_neg_anchors}
        if self.log:
            wandb.log(log)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels != -1
        # prit((gt_targets[valid_mask]>0.7).sum())
        # print( cat(pred_objectness_logits, dim=1)[valid_mask],gt_labels[valid_mask])
        pred_objectness_logits = { key: cat(pred_objectness_logits[key],dim=2)
                        for key in pred_objectness_logits}
        # print(gt_labels.shape,pred_objectness_logits['pi'].shape)
        # objectness_loss = smooth_l1_loss(
        #     torch.sigmoid(cat(pred_objectness_logits, dim=1)[valid_mask]),
        #     gt_targets[valid_mask].to(torch.float32),beta=self.smooth_l1_beta,
        #     reduction="sum",
        # )
        objectness_loss = mdn_loss(pred_objectness_logits,valid_mask,
                            gt_targets[valid_mask].to(torch.float32)).sum()
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.
        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_objectness_logits = find_logit(pred_objectness_logits)
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def predict_proposals_uncertainty(self,anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],):
        with torch.no_grad():
            objectness_uncertainty = mdn_uncertainties(pred_objectness_logits) # [N x total anchors]
            pred_objectness_logits = find_logit(pred_objectness_logits)
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals_uncertainty(
                pred_proposals, objectness_uncertainty,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals