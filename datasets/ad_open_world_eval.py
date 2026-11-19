
from collections import defaultdict
import copy

import numpy as np
import torch

from util.misc import all_gather


def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            ap += (np.max(prec[rec >= t]) if np.sum(rec >= t) else 0) / 11.0
        return ap

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def voc_evaluate(detections, annotations, cid, ovthresh=0.5, use_07_metric=True):
    class_gt_bboxes = {}
    npos = 0
    for image_id, ann in annotations.items():
        ann = np.asarray(ann, dtype=np.float32).reshape(-1, 5)
        if cid == -1:
            rows = ann[:, :4]
        else:
            rows = np.asarray([obj[:4] for obj in ann if int(obj[-1]) == cid])
        difficult = np.zeros(len(rows), dtype=bool)
        class_gt_bboxes[image_id] = {
            "bbox": rows.reshape(-1, 4),
            "difficult": difficult,
            "det": [False] * len(rows),
        }
        npos += int(np.sum(~difficult))

    det_image_ids = []
    confidences = []
    det_bboxes = []
    for image_id, dets in detections.get(cid, {}).items():
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
        for det in dets:
            det_image_ids.append(image_id)
            det_bboxes.append(det[:4])
            confidences.append(det[-1])

    if len(det_image_ids) == 0 or npos == 0:
        return 0.0, 0.0, 0.0

    det_bboxes = np.asarray(det_bboxes, dtype=np.float32).reshape(-1, 4)
    confidences = np.asarray(confidences, dtype=np.float32)
    sorted_ind = np.argsort(-confidences)
    det_bboxes = det_bboxes[sorted_ind]
    det_image_ids = [det_image_ids[i] for i in sorted_ind]

    tp = np.zeros(len(det_image_ids))
    fp = np.zeros(len(det_image_ids))
    for d, image_id in enumerate(det_image_ids):
        record = class_gt_bboxes[image_id]
        bb = det_bboxes[d].astype(float)
        gt = record["bbox"].astype(float)
        ovmax = -np.inf
        jmax = -1
        if gt.size > 0:
            ixmin = np.maximum(gt[:, 0], bb[0])
            iymin = np.maximum(gt[:, 1], bb[1])
            ixmax = np.minimum(gt[:, 2], bb[2])
            iymax = np.minimum(gt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih
            union = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (gt[:, 2] - gt[:, 0] + 1.0) * (gt[:, 3] - gt[:, 1] + 1.0)
                - inters
            )
            overlaps = inters / np.maximum(union, np.finfo(np.float64).eps)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh and not record["difficult"][jmax]:
            if not record["det"][jmax]:
                tp[d] = 1.0
                record["det"][jmax] = True
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    recall = float(np.sum(tp) / npos)
    precision = float(np.sum(tp) / max(np.sum(tp) + np.sum(fp), np.finfo(np.float64).eps))
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = float(voc_ap(rec, prec, use_07_metric))
    return recall, precision, ap


class ADOWEvaluator:
    def __init__(self, dataset, iou_types, args=None):
        assert tuple(iou_types) == ("bbox",)
        self.dataset = dataset
        self.class_names = dataset.CLASS_NAMES
        self.num_known_classes = getattr(dataset, "num_known_classes", len(self.class_names) - 1)
        self.unknown_class_index = getattr(dataset, "unknown_class_index", len(self.class_names) - 1)
        self.predictions = defaultdict(list)
        self.coco_eval = {"bbox": lambda: None}
        self.coco_eval["bbox"].stats = torch.tensor([])
        self.coco_eval["bbox"].eval = {}
        self.metrics = {}

    def update(self, predictions):
        for image_id, pred in predictions.items():
            boxes = pred["boxes"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            for box, label, score in zip(boxes, labels, scores):
                self.predictions[int(image_id)].append(
                    [float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(label), float(score)]
                )

    def synchronize_between_processes(self):
        gathered = all_gather(dict(self.predictions))
        merged = defaultdict(list)
        for part in gathered:
            for image_id, rows in part.items():
                merged[int(image_id)].extend(rows)
        self.predictions = merged

    def _detections_for_class(self, cid, select_num=100):
        result = {cid: {}}
        for image_id in self.dataset.ids:
            rows = self.predictions.get(int(image_id), [])
            if cid == -1:
                selected = rows
            else:
                selected = [r for r in rows if int(r[4]) == cid]
            selected = sorted(selected, key=lambda r: r[5], reverse=True)[:select_num]
            result[cid][int(image_id)] = np.asarray(
                [[r[0], r[1], r[2], r[3], r[5]] for r in selected],
                dtype=np.float32,
            ).reshape(-1, 5)
        return result

    def accumulate(self):
        gt = {
            int(image_id): np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            for image_id, rows in self.dataset.eval_gt.items()
        }
        ap_list = []
        recall_list = []
        precision_list = []
        for cid in list(range(self.num_known_classes)) + [self.unknown_class_index, -1]:
            detections = self._detections_for_class(cid, select_num=100)
            recall, precision, ap = voc_evaluate(detections, copy.deepcopy(gt), cid)
            ap_list.append(ap)
            recall_list.append(recall)
            precision_list.append(precision)

        unknown_recalls = {}
        for k in (10, 20, 30):
            detections = self._detections_for_class(self.unknown_class_index, select_num=k)
            recall, _, _ = voc_evaluate(
                detections, copy.deepcopy(gt), self.unknown_class_index
            )
            unknown_recalls[k] = recall

        known_ap = float(np.mean(ap_list[: self.num_known_classes]))
        self.metrics = {
            "known_ap": known_ap,
            "unknown_ap": ap_list[-2],
            "agnostic_ap": ap_list[-1],
            "agnostic_recall": recall_list[-1],
            "unknown_recall": recall_list[-2],
            "unknown_recall@10": unknown_recalls[10],
            "unknown_recall@20": unknown_recalls[20],
            "unknown_recall@30": unknown_recalls[30],
        }
        self.coco_eval["bbox"].stats = torch.tensor(
            [
                self.metrics["known_ap"],
                self.metrics["unknown_ap"],
                self.metrics["agnostic_ap"],
                self.metrics["unknown_recall"],
            ],
            dtype=torch.float32,
        )
        self.coco_eval["bbox"].eval = self.metrics

    def summarize(self):
        print("AD-OWOD known / CODA-unknown metrics")
        for key, value in self.metrics.items():
            print(f"{key}: {value:.6f}")
        return dict(self.metrics)
