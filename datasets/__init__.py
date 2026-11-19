

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file in ['owod', 'towod', 'owdetr', 'voc2007']:
        from .torchvision_datasets.open_world import build as build_owod
        return build_owod(image_set, args)
    if args.dataset_file in ['soda_coda', 'ad_owod_soda', 'soda']:
        from .ad_owod import build_soda_coda_datasets
        dataset_train, dataset_val = build_soda_coda_datasets(args)
        return dataset_train if 'train' in image_set else dataset_val
    if args.dataset_file in ['bdd_coda', 'ad_owod_bdd', 'bdd']:
        from .ad_owod import build_bdd_coda_datasets
        dataset_train, dataset_val = build_bdd_coda_datasets(args)
        return dataset_train if 'train' in image_set else dataset_val
    raise ValueError(f'dataset {args.dataset_file} not supported')
