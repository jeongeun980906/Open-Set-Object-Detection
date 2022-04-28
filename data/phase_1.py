# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

# from data import DatasetCatalog, MetadataCatalog
from structures.box import BoxMode
from tools_det.fileio import PathManager


VOC_CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
COCO_CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"

)

VOC_CLASS_NAMES_COCOFIED = [
        "airplane", "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]
BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant", "sofa", "tvmonitor"
    ]

def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]
                    ,phase='t1',COCO_CLASS=False):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    if phase is not None:
        with PathManager.open(os.path.join(dirname, "ImageSets", "Main", '{}_{}.txt'.format(phase,split))) as f:
            fileids = np.loadtxt(f, dtype=np.str)
    else:
        if split == 'test':
            path = os.path.join('/data/jeongeun/OWOD_datasets','OWOD_imagesets', "all_task_{}.txt".format(split))
            with PathManager.open(path) as f:
                fileids = np.loadtxt(f, dtype=np.str)
        else:
            phase = ['t1','t2','t3','t4']
            fileids = []
            for p in phase:
                with PathManager.open(os.path.join(dirname, "ImageSets", "Main", '{}_{}.txt'.format(p,split))) as f:
                    fileid = np.loadtxt(f, dtype=np.str)
                fileids.extend(fileid)
    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in class_names:
                # We include "difficult" samples in training.
                # Based on limited experiments, they don't hurt accuracy.
                # difficult = int(obj.find("difficult").text)
                # if difficult == 1:
                # continue
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                # Original annotations are integers in the range [1, W or H]
                # Assuming they mean 1-based pixel indices (inclusive),
                # a box with annotation (xmin=1, xmax=W) covers the whole image.
                # In coordinate space this is represented by (xmin=0, xmax=W)
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                instances.append(
                    {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS})
            
            else:
                if cls in VOC_CLASS_NAMES_COCOFIED:
                    ncls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
                    bbox = obj.find("bndbox")
                    bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    # Original annotations are integers in the range [1, W or H]
                    # Assuming they mean 1-based pixel indices (inclusive),
                    # a box with annotation (xmin=1, xmax=W) covers the whole image.
                    # In coordinate space this is represented by (xmin=0, xmax=W)
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0
                    instances.append(
                        {"category_id": class_names.index(ncls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS})
                elif COCO_CLASS:
                    bbox = obj.find("bndbox")
                    bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    # Original annotations are integers in the range [1, W or H]
                    # Assuming they mean 1-based pixel indices (inclusive),
                    # a box with annotation (xmin=1, xmax=W) covers the whole image.
                    # In coordinate space this is represented by (xmin=0, xmax=W)
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0
                    instances.append(
                        {"category_id": len(class_names), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS})
        r["annotations"] = instances
        dicts.append(r)
    return dicts


# def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
#     DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
#     MetadataCatalog.get(name).set(
#         thing_classes=list(class_names), dirname=dirname, year=year, split=split
#     )

