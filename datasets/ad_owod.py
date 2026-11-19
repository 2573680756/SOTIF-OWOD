
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .ad_categories import (
    BDD_CATEGORY_NAME_TO_LABEL,
    BDD_CODA_CLASS_NAMES,
    CODA_CATEGORY_ID_TO_BDD_LABEL,
    CODA_CATEGORY_ID_TO_SODA_LABEL,
    SODA_CATEGORY_ID_TO_LABEL,
    SODA_CODA_CLASS_NAMES,
)
from .coco import make_coco_transforms


class ADOWODDetection(Dataset):

    def __init__(
        self,
        ann_file,
        img_folder,
        image_set,
        transforms=None,
        class_names=SODA_CODA_CLASS_NAMES,
        train_category_id_to_label=SODA_CATEGORY_ID_TO_LABEL,
        eval_category_id_to_label=CODA_CATEGORY_ID_TO_SODA_LABEL,
    ):
        self.ann_file = str(ann_file)
        self.img_folder = str(img_folder)
        self.image_set = image_set
        self.transforms = transforms
        self.CLASS_NAMES = tuple(class_names)
        self.train_category_id_to_label = dict(train_category_id_to_label)
        self.eval_category_id_to_label = dict(eval_category_id_to_label)
        self.num_known_classes = len(self.CLASS_NAMES) - 1
        self.unknown_class_index = len(self.CLASS_NAMES) - 1
        self.is_train = "train" in image_set

        with open(self.ann_file, "r") as f:
            data = json.load(f)

        self.images = list(data.get("images", []))
        self.categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        self.annotations_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            self.annotations_by_image[ann["image_id"]].append(ann)

        self.ids = [img["id"] for img in self.images]
        self.image_info = {img["id"]: img for img in self.images}
        self.eval_gt = self._build_eval_gt()

    def __len__(self):
        return len(self.images)

    def _label_from_category(self, category_id):
        if self.is_train:
            return self.train_category_id_to_label.get(category_id)
        return self.eval_category_id_to_label.get(
            category_id, self.unknown_class_index
        )

    def _image_path(self, image_info):
        file_name = image_info["file_name"]
        path = Path(self.img_folder) / file_name
        if path.exists():
            return path

        alt = Path(self.img_folder) / "images" / file_name
        if alt.exists():
            return alt
        return path

    def _annotation_to_instance(self, ann, width, height):
        label = self._label_from_category(ann["category_id"])
        if label is None:
            return None

        x, y, w, h = ann["bbox"]
        x0 = max(float(x), 0.0)
        y0 = max(float(y), 0.0)
        x1 = min(float(x + w), float(width))
        y1 = min(float(y + h), float(height))
        if x1 <= x0 or y1 <= y0:
            return None

        return {
            "bbox": [x0, y0, x1, y1],
            "label": label,
            "area": float(ann.get("area", (x1 - x0) * (y1 - y0))),
            "iscrowd": int(ann.get("iscrowd", 0)),
        }

    def _instances_for_image(self, image_info):
        width = int(image_info["width"])
        height = int(image_info["height"])
        instances = []
        for ann in self.annotations_by_image.get(image_info["id"], []):
            instance = self._annotation_to_instance(ann, width, height)
            if instance is not None:
                instances.append(instance)
        return instances

    def _build_eval_gt(self):
        gt = {}
        for image_info in self.images:
            instances = self._instances_for_image(image_info)
            rows = [inst["bbox"] + [inst["label"]] for inst in instances]
            gt[image_info["id"]] = rows
        return gt

    def __getitem__(self, index):
        image_info = self.images[index]
        img_path = self._image_path(image_info)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        instances = self._instances_for_image(image_info)
        boxes = torch.as_tensor([i["bbox"] for i in instances], dtype=torch.float32)
        boxes = boxes.reshape(-1, 4)

        target = {
            "image_id": torch.tensor([image_info["id"]], dtype=torch.int64),
            "labels": torch.tensor([i["label"] for i in instances], dtype=torch.int64),
            "area": torch.tensor([i["area"] for i in instances], dtype=torch.float32),
            "boxes": boxes,
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
            "iscrowd": torch.tensor([i["iscrowd"] for i in instances], dtype=torch.uint8),
        }

        if self.transforms is not None:
            img, target = self.transforms[-1](img, target)
        return img, target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(image_set={self.image_set!r}, "
            f"images={len(self.images)}, ann_file={os.path.relpath(self.ann_file)!r})"
        )


def build_soda_coda_datasets(args):
    train_ann = getattr(args, "ad_train_ann", None) or os.path.join(
        args.data_root, "annotations", "train.json"
    )
    train_img_dir = getattr(args, "ad_train_img_dir", None) or os.path.join(
        args.data_root, "train"
    )
    val_ann = getattr(args, "ad_val_ann", None) or os.path.join(
        args.data_root, "annotations", "annotations.json"
    )
    val_img_dir = getattr(args, "ad_val_img_dir", None) or os.path.join(
        args.data_root, "val", "images"
    )

    dataset_train = ADOWODDetection(
        train_ann,
        train_img_dir,
        image_set=getattr(args, "train_set", "soda_train"),
        transforms=make_coco_transforms(getattr(args, "train_set", "soda_train")),
    )
    dataset_val = ADOWODDetection(
        val_ann,
        val_img_dir,
        image_set=getattr(args, "test_set", "coda_val"),
        transforms=make_coco_transforms(getattr(args, "test_set", "coda_val")),
    )
    return dataset_train, dataset_val


class BDDRawDetection(Dataset):

    def __init__(
        self,
        ann_file,
        img_folder,
        image_set,
        transforms=None,
        class_names=BDD_CODA_CLASS_NAMES,
        category_name_to_label=BDD_CATEGORY_NAME_TO_LABEL,
        image_width=1280,
        image_height=720,
        filter_empty=True,
    ):
        self.ann_file = str(ann_file)
        self.img_folder = str(img_folder)
        self.image_set = image_set
        self.transforms = transforms
        self.CLASS_NAMES = tuple(class_names)
        self.category_name_to_label = dict(category_name_to_label)
        self.num_known_classes = len(self.CLASS_NAMES) - 1
        self.unknown_class_index = len(self.CLASS_NAMES) - 1
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        with open(self.ann_file, "r") as f:
            raw_images = json.load(f)

        self.images = []
        self.raw_labels = {}
        next_id = 1
        for raw in raw_images:
            image_info = {
                "id": next_id,
                "file_name": raw["name"],
                "width": self.image_width,
                "height": self.image_height,
            }
            labels = raw.get("labels", [])
            self.raw_labels[next_id] = labels
            if filter_empty and not self._instances_for_image(image_info):
                next_id += 1
                continue
            self.images.append(image_info)
            next_id += 1

        self.ids = [img["id"] for img in self.images]
        self.image_info = {img["id"]: img for img in self.images}
        self.eval_gt = self._build_eval_gt()

    def __len__(self):
        return len(self.images)

    def _image_path(self, image_info):
        return Path(self.img_folder) / image_info["file_name"]

    def _annotation_to_instance(self, ann, width, height):
        if "box2d" not in ann:
            return None
        label = self.category_name_to_label.get(ann.get("category"))
        if label is None:
            return None

        box = ann["box2d"]
        x0 = max(float(box["x1"]), 0.0)
        y0 = max(float(box["y1"]), 0.0)
        x1 = min(float(box["x2"]), float(width))
        y1 = min(float(box["y2"]), float(height))
        if x1 <= x0 or y1 <= y0:
            return None

        return {
            "bbox": [x0, y0, x1, y1],
            "label": label,
            "area": float((x1 - x0) * (y1 - y0)),
            "iscrowd": 0,
        }

    def _instances_for_image(self, image_info, width=None, height=None):
        width = int(width or image_info["width"])
        height = int(height or image_info["height"])
        instances = []
        for ann in self.raw_labels.get(image_info["id"], []):
            instance = self._annotation_to_instance(ann, width, height)
            if instance is not None:
                instances.append(instance)
        return instances

    def _build_eval_gt(self):
        gt = {}
        for image_info in self.images:
            instances = self._instances_for_image(image_info)
            rows = [inst["bbox"] + [inst["label"]] for inst in instances]
            gt[image_info["id"]] = rows
        return gt

    def __getitem__(self, index):
        image_info = self.images[index]
        img_path = self._image_path(image_info)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        instances = self._instances_for_image(image_info, width, height)
        boxes = torch.as_tensor([i["bbox"] for i in instances], dtype=torch.float32)
        boxes = boxes.reshape(-1, 4)

        target = {
            "image_id": torch.tensor([image_info["id"]], dtype=torch.int64),
            "labels": torch.tensor([i["label"] for i in instances], dtype=torch.int64),
            "area": torch.tensor([i["area"] for i in instances], dtype=torch.float32),
            "boxes": boxes,
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
            "iscrowd": torch.tensor([i["iscrowd"] for i in instances], dtype=torch.uint8),
        }

        if self.transforms is not None:
            img, target = self.transforms[-1](img, target)
        return img, target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(image_set={self.image_set!r}, "
            f"images={len(self.images)}, ann_file={os.path.relpath(self.ann_file)!r})"
        )


def build_bdd_coda_datasets(args):
    train_ann = getattr(args, "ad_train_ann", None) or os.path.join(
        args.data_root, "annotations", "bdd100k_labels_images_train.json"
    )
    train_img_dir = getattr(args, "ad_train_img_dir", None) or os.path.join(
        args.data_root, "train", "images"
    )
    val_ann = getattr(args, "ad_val_ann", None) or os.path.join(
        args.data_root, "annotations", "annotations.json"
    )
    val_img_dir = getattr(args, "ad_val_img_dir", None) or os.path.join(
        args.data_root, "val", "images"
    )

    dataset_train = BDDRawDetection(
        train_ann,
        train_img_dir,
        image_set=getattr(args, "train_set", "bdd_train"),
        transforms=make_coco_transforms(getattr(args, "train_set", "bdd_train")),
    )
    dataset_val = ADOWODDetection(
        val_ann,
        val_img_dir,
        image_set=getattr(args, "test_set", "coda_val"),
        transforms=make_coco_transforms(getattr(args, "test_set", "coda_val")),
        class_names=BDD_CODA_CLASS_NAMES,
        train_category_id_to_label={},
        eval_category_id_to_label=CODA_CATEGORY_ID_TO_BDD_LABEL,
    )
    return dataset_train, dataset_val
