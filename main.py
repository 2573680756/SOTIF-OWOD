

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from engine import evaluate, train_one_epoch



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='base learning rate for the Transformer and prediction heads')
    
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+',
                        help='list of backbone layer names, used to assign a different learning rate')
    
    parser.add_argument('--lr_backbone', default=2e-6, type=float,
                        help='learning rate for the backbone, usually 1/10 of the base learning rate')
    
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], 
                        type=str, nargs='+',
                        help='list of linear projection layer names for deformable attention sampling offsets and reference points')
    
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float,
                        help='learning rate multiplier for linear projection layers, used to stabilize training')
    
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size per GPU; total batch size = batch_size * num_gpus')
    
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay coefficient for L2 regularization')
    
    parser.add_argument('--epochs', default=70, type=int,
                        help='total number of training epochs')
    
    parser.add_argument('--lr_drop', default=20, type=int,
                        help='epoch interval for learning rate decay')
    
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+',
                        help='specific epochs at which to decay the learning rate, e.g. [40, 60]')
    
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='maximum norm for gradient clipping, prevents gradient explosion')

    parser.add_argument('--sgd', action='store_true',
                        help='use SGD optimizer instead of AdamW')

    parser.add_argument('--with_box_refine', default=True, action='store_true',
                        help='enable iterative bounding box refinement; each layer predicts offsets w.r.t. the previous layer')
    
    parser.add_argument('--two_stage', default=True, action='store_true',
                        help='enable two-stage mode, using encoder outputs as region proposals')

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='path to the pretrained model; when set only the segmentation head is trained')

    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='backbone type, e.g. resnet50, resnet101')
    
    parser.add_argument('--dilation', action='store_true',
                        help='use dilated convolution instead of stride in the last conv block to increase feature resolution')
    
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help='position embedding type: sine (sinusoidal) or learned')
    
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help='scale factor for position embedding: position / size * scale')
    
    parser.add_argument('--num_feature_levels', default=4, type=int, 
                        help='number of multi-scale feature levels, usually 4')

    parser.add_argument('--enc_layers', default=6, type=int,
                        help='number of Transformer encoder layers')
    
    parser.add_argument('--dec_layers', default=6, type=int,
                        help='number of Transformer decoder layers')
    
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help='hidden dimension of the FFN in Transformer')
    
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='hidden dimension of Transformer, the unified dimension of all features')
    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate in Transformer')
    
    parser.add_argument('--nheads', default=8, type=int,
                        help='number of heads in multi-head attention')
    
    parser.add_argument('--num_queries', default=300, type=int,
                        help='number of object queries, i.e. the maximum number of detected objects')
    
    parser.add_argument('--dec_n_points', default=8, type=int,
                        help='number of sampling points per head in decoder deformable attention')
    
    parser.add_argument('--enc_n_points', default=8, type=int,
                        help='number of sampling points per head in encoder deformable attention')

    parser.add_argument('--masks', action='store_true',
                        help='train the segmentation head for instance or panoptic segmentation')

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help='disable auxiliary losses, i.e. the losses of intermediate layers')

    parser.add_argument('--set_cost_class', default=2, type=float,
                        help='weight of the class cost in Hungarian matching')
    
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help='weight of the bounding box L1 cost in Hungarian matching')
    
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help='weight of the GIoU cost in Hungarian matching')

    parser.add_argument('--mask_loss_coef', default=1, type=float,
                        help='weight of the mask focal loss')
    
    parser.add_argument('--dice_loss_coef', default=1, type=float,
                        help='weight of the Dice loss')
    
    parser.add_argument('--cls_loss_coef', default=2, type=float,
                        help='weight of the classification loss (focal loss)')
    
    parser.add_argument('--bbox_loss_coef', default=5, type=float,
                        help='weight of the bounding box L1 loss')
    
    parser.add_argument('--giou_loss_coef', default=2, type=float,
                        help='weight of the GIoU loss')
    
    parser.add_argument('--obj_loss_coef', default=3, type=float,
                        help='weight of the objectness loss (UN-DETR specific)')
    
    parser.add_argument('--f_obj_loss_coef', default=3, type=float,
                        help='weight of the first-layer objectness loss')
    
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='alpha parameter of focal loss, handles class imbalance')

    parser.add_argument('--dataset', default='TOWOD',
                        help='dataset type, options: coco, coco_panoptic, owod, OWDETR, TOWOD, VOC2007, soda_coda, bdd_coda')

    parser.add_argument('--dataset_file', default=None,
                        help='kept for the legacy build_dataset interface; defaults to --dataset')

    parser.add_argument('--coco_path', default='/media/cyy/cyy2/zwy/UN-DETR-master/data', type=str,
                        help='root directory of the dataset, supports COCO and OWOD')

    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int,
                        help='number of previously introduced classes, for incremental learning')

    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int,
                        help='number of classes introduced in the current task, for incremental learning')

    parser.add_argument('--image_set', default='owdetr_t1_train', type=str,
                        help='name of the dataset split file, e.g. train, val, test, voc2007_trainval')
    
    parser.add_argument('--train_set', default='owod_t1_train',
                       help='training set file list (txt)')
    
    parser.add_argument('--test_set', default='owod_all_task_test',
                       help='test set file list (txt)')
    
    parser.add_argument('--data_root', default='./data/OWOD', type=str,
                    help='root directory of the dataset')

    parser.add_argument('--ad_train_ann', default=None, type=str,
                        help='path to the autonomous driving training annotations, supports COCO JSON or BDD raw JSON')

    parser.add_argument('--ad_train_img_dir', default=None, type=str,
                        help='directory of the autonomous driving training images')

    parser.add_argument('--ad_val_ann', default=None, type=str,
                        help='path to the autonomous driving validation/test annotations (COCO JSON)')

    parser.add_argument('--ad_val_img_dir', default=None, type=str,
                        help='directory of the autonomous driving validation/test images')
    
    parser.add_argument('--coco_panoptic_path', type=str,
                        help='path to the COCO Panoptic dataset, for panoptic segmentation')
    
    parser.add_argument('--remove_difficult', action='store_true',
                        help='remove samples marked as difficult')

    parser.add_argument('--output_dir', default='./output_owdetr_t1',
                        help='path to save models and logs; an empty string means nothing is saved')
    
    parser.add_argument('--device', default='cuda',
                        help='device used for training and testing, cuda or cpu')
    
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed, ensures reproducibility')
    
    parser.add_argument('--resume', default='./output_bdd_coda/checkpoint.pth', 
                        help='path to the checkpoint to resume training from')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='starting epoch, used when resuming training')
    
    parser.add_argument('--eval', action='store_true',
                        help='only run evaluation without training')
    
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of data loader worker processes')
    
    parser.add_argument('--cache_mode', default=False, action='store_true', 
                        help='cache images in memory to speed up data loading')

    parser.add_argument('--pretrained', type=str, default='',
                        help='path to the pretrained weights used for initialization')
    
    parser.add_argument('--num_classes', default=81, type=int,
                        help='total number of classes; SODA+CODA is 6 known + 1 unknown = 7, BDD+CODA is 5 known + 1 unknown = 6')

    parser.add_argument('--num_known_classes', default=20, type=int,
                        help='number of known classes; the unknown class is placed at num_classes-1 by default')

    parser.add_argument('--obj_threshold', default=0.2, type=float,
                        help='objectness pre-filter threshold in post-processing')

    parser.add_argument('--known_threshold', default=0.1, type=float,
                        help='score threshold for known classes in post-processing')

    parser.add_argument('--unknown_threshold', default=0.1, type=float,
                        help='score threshold for unknown classes in post-processing')

    parser.add_argument('--known_score_weight', default=2.0, type=float,
                        help='score scaling factor for known classes in post-processing')

    parser.add_argument('--unknown_score_weight', default=0.45, type=float,
                        help='score scaling factor for unknown classes in post-processing')

    parser.add_argument('--post_topk', default=300, type=int,
                        help='number of candidate boxes kept per image before NMS in post-processing')

    parser.add_argument('--nms_iou_threshold', default=0.7, type=float,
                        help='IoU threshold for NMS in post-processing')

    parser.add_argument('--known_unknown_nms_iou_threshold', default=1, type=float,
                        help='IoU threshold for cross-class NMS between known and unknown classes; follows --nms_iou_threshold by default, keeping known and removing unknown on overlap')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    if args.dataset_file is None:
        args.dataset_file = args.dataset
    
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    
    model.to(device)

    model_without_ddp = model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train, dataset_val = get_datasets(args)
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) 
                 and not match_name_keywords(n, args.lr_linear_proj_names) 
                 and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                      if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                      if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    
    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, 
            lr=args.lr, 
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    lr_milestones = args.lr_drop_epochs if args.lr_drop_epochs is not None else [25, 50]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_milestones
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    if args.dataset == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
        
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        
        print(checkpoint['model'].keys())
        for key in checkpoint['model'].keys():
            print(f"Key: {key}, Shape: {checkpoint['model'][key].shape}")

        target_shape = torch.Size([300, 512])
        for key, value in checkpoint['model'].items():
            if value.shape == target_shape:
                print(f"Key with shape {target_shape}: {key}")
        
        checkpoint['model'].pop('query_embed.weight')
        
      
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        
        unexpected_keys = [k for k in unexpected_keys 
                          if not (k.endswith('total_params') or k.endswith('total_ops'))]
        
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        
        
    
    
    output_dir = Path(args.output_dir)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        
        unexpected_keys = [k for k in unexpected_keys 
                          if not (k.endswith('total_params') or k.endswith('total_ops'))]
        
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            
            p_groups = copy.deepcopy(optimizer.param_groups)
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            
            print(optimizer.param_groups)
            
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, '
                      'so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], 
                                                 optimizer.param_groups))
            
            lr_scheduler.step(lr_scheduler.last_epoch)
            
            args.start_epoch = checkpoint['epoch'] + 1
        
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors,
            data_loader_val, base_ds, device, args.output_dir,args)
        
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, 
                                output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, 
            args.clip_max_norm)
        
        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            if epoch % 5 == 0:
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors,
                    data_loader_val, base_ds, device, args.output_dir, args
                )
            if (epoch) % args.lr_drop == 0 or (epoch) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)


        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if epoch %5==0:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def get_datasets(args):
    print(args.dataset)

    if args.dataset.lower() in ['soda_coda', 'ad_owod_soda', 'soda']:
        from datasets.ad_owod import build_soda_coda_datasets
        dataset_train, dataset_val = build_soda_coda_datasets(args)
        print(args.train_set)
        print(args.test_set)
        print(dataset_train)
        print(dataset_val)
        return dataset_train, dataset_val

    if args.dataset.lower() in ['bdd_coda', 'ad_owod_bdd', 'bdd']:
        from datasets.ad_owod import build_bdd_coda_datasets
        dataset_train, dataset_val = build_bdd_coda_datasets(args)
        print(args.train_set)
        print(args.test_set)
        print(dataset_train)
        print(dataset_val)
        return dataset_train, dataset_val
    
    dataset_train = OWDetection(
        args,
        args.data_root,
        image_set=args.train_set,
        transforms=make_coco_transforms(args.train_set),
        dataset=args.dataset
    )
    
    dataset_val = OWDetection(
        args,
        args.data_root,
        image_set=args.test_set,
        dataset=args.dataset,
        transforms=make_coco_transforms(args.test_set)
    )

    print(args.train_set)
    print(args.test_set)
    print(dataset_train)
    print(dataset_val)

    return dataset_train, dataset_val


def create_ft_dataset(args, image_sorted_scores):
    print(f'found a total of {len(image_sorted_scores.keys())} images')
    
    tmp_dir = args.data_root + '/ImageSets/' + args.dataset + "/" + args.exemplar_replay_dir + "/"

    class_sorted_scores = {}
    imgs_per_class = {}
    
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS + args.PREV_INTRODUCED_CLS):
        class_sorted_scores[str(i)] = []
        imgs_per_class[str(i)] = []

    for k, v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            class_sorted_scores[str(v['labels'][j])].append(v['scores'][j])

    class_threshold = {}
    
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS + args.PREV_INTRODUCED_CLS):
        tmp = np.array(class_sorted_scores[str(i)])
        tmp.sort()
        
        if len(tmp) > args.num_inst_per_class and not args.exemplar_replay_random:
            max_val = tmp[-args.num_inst_per_class // 2]
            min_val = tmp[args.num_inst_per_class // 2]
        else:
            if args.exemplar_replay_random:
                print('using random exemplar selection')
            else:
                print(f'only found {len(tmp)} imgs in class {i}')
            max_val = tmp.min()
            min_val = tmp.max()
            
        class_threshold[str(i)] = (min_val, max_val)

    save_imgs = []
    
    for k, v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            label = str(v['labels'][j])
            score = v['scores'][j]
            
            if (score <= class_threshold[label][0].numpy() or score >= class_threshold[label][1].numpy())and (len(imgs_per_class[label]) <= args.num_inst_per_class + 2):
                save_imgs.append(k)
                imgs_per_class[label].append(k)
                        
    print(f'found {len(np.unique(save_imgs))} images in run')
    
    if len(args.exemplar_replay_prev_file) > 0:
        previous_ft = open(tmp_dir + args.exemplar_replay_prev_file, 'r').read().splitlines()
        save_imgs += previous_ft
        
    save_imgs = np.unique(save_imgs)
    np.random.shuffle(save_imgs)
    
    if len(save_imgs) > args.exemplar_replay_max_length:
        save_imgs = save_imgs[:args.exemplar_replay_max_length]
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    with open(tmp_dir + args.exemplar_replay_cur_file, 'w') as f:
        for line in save_imgs:
            f.write(line)
            f.write('\n')
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', 
        parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
