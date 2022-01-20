import torch
from layers.nms import batched_nms

import data.transforms as T

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg,model):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = model
        self.model.eval()
        # if len(cfg.DATASETS.TEST):
        #     self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
    
    def proposal_predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            images = self.model.preprocess_image([inputs])
            features = self.model.backbone(images.tensor)
            self.model.proposal_generator.nms_thresh = 0.1
            proposals, _ = self.model.proposal_generator(images, features, None)
            proposals = self.model._postprocess(proposals,[inputs],images.image_sizes)
            return proposals

    def uncertainty_predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            images = self.model.preprocess_image([inputs])
            features = self.model.backbone(images.tensor)
            # extract feature
            features = [features[f] for f in self.model.proposal_generator.in_features]
            anchors = self.model.proposal_generator.anchor_generator(features)
            pred_objectness_logits, pred_anchor_deltas = self.model.proposal_generator.head(features)

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
                x.view(x.shape[0], -1, self.model.proposal_generator.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
                for x in pred_anchor_deltas
            ]
            proposals = self.model.proposal_generator.predict_proposals_uncertainty(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

            proposals = self.model._postprocess(proposals,[inputs],images.image_sizes)
            return proposals