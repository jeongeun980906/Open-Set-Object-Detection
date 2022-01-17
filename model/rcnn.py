from model.backbone.resnet import build_backbone
from model.rpn.rpn import build_proposal_genreator
from model.roi_heads.roi_heads import build_roi_heads
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from structures.image_list import ImageList
from structures.instances import Instances
from model.postprocess import detector_postprocess

class GeneralizedRCNN(nn.Module):
    def __init__(self,cfg,device='cuda'):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_genreator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        self.input_format = cfg.INPUT.FORMAT
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(device)
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(device)
        self.device = device

    def forward(self,batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def preprocess_image(self,batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]
            detected_instances (None or list[Instances]): `Instances`
                object contains "pred_boxes" and "pred_classes"
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, nms
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results