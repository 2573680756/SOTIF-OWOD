
import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from torchvision.ops.boxes import batched_nms


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ObjectHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0, 1)
        
        self.linear2 = nn.Linear(hidden_dim, 1)
        
        self.activate = nn.Sigmoid()
        
        nn.init.constant_(self.linear2.weight, 0)
        nn.init.constant_(self.linear2.bias, 0)

       
    def fressze_obj_head(self):
        self.obj_head.eval()
    
    def forward(self, x):
        out = self.flatten(x)
        
        out = self.linear2(out)
        
        out = self.activate(out)
        
        out = out.unflatten(0, x.shape[:2])
        return out




class DeformableDETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.num_feature_levels = num_feature_levels
        
        self.object_head = ObjectHead(hidden_dim)
        
        if not two_stage:
            self.query_embed_cls = nn.Embedding(num_queries, hidden_dim)
            self.query_embed_loc = nn.Embedding(num_queries, hidden_dim)
        
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.object_head = _get_clones(self.object_head, num_pred)
        
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.object_head = nn.ModuleList([self.object_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            self.transformer.decoder.object_head = self.object_head
            
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
            
            

    def forward(self, samples: NestedTensor):

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)
       
        srcs = []
        masks = []
        
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds_cls = None
        query_embeds_loc = None
        if not self.two_stage:
            query_embeds_cls = self.query_embed_cls.weight
            query_embeds_loc = self.query_embed_loc.weight
        
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, enc_outputs_obj = self.transformer(srcs, masks, pos, query_embeds_cls, query_embeds_loc)

        outputs_classes = []
        outputs_coords = []
        outputs_object = []
      
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            
            hs_cls = hs[lvl, 0]
            hs_loc = hs[lvl, 1]
            
            outputs_class = self.class_embed[lvl](hs_cls)
            outputs_obj = self.object_head[lvl](hs_loc)
            
            tmp = self.bbox_embed[lvl](hs_loc)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_object.append(outputs_obj)
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_obj = torch.stack(outputs_object)
        

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'obj': outputs_obj[-1]
        }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_obj)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
         
            out['enc_outputs'] = {
                'pred_logits': enc_outputs_class,
                'pred_boxes': enc_outputs_coord,
                'obj': enc_outputs_obj
            }
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_object):
        return [{'pred_logits': a, 'pred_boxes': b, 'obj': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_object[:-1])]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 num_known_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_known_classes = num_known_classes or max(num_classes - 1, 1)
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
    
    def loss_labels(self, outputs, targets, indices, second_indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, second_indices,  num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, second_indices,  num_boxes):
        assert 'pred_boxes' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def calc_objectness(self, src_bbox, gt_bbox):
        c_x, c_y = src_bbox[:2]
        
        l, t, r, b = gt_bbox[0] - gt_bbox[2] / 2, gt_bbox[1] - gt_bbox[3] / 2, \
                     gt_bbox[0] + gt_bbox[2] / 2, gt_bbox[1] + gt_bbox[3] / 2
       
        l_d = torch.abs(c_x - l)
        t_d = torch.abs(c_y - t)
        r_d = torch.abs(r - c_x)
        b_d = torch.abs(b - c_y)
       
        centerness = torch.sqrt((torch.min(l_d, r_d) / torch.max(l_d, r_d)) * 
                               (torch.min(t_d, b_d) / torch.max(t_d, b_d)))
        return centerness
    
    def loss_obj(self, outputs, targets, indices, second_indices, num_boxes):
        assert 'obj' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        src_logits = torch.sigmoid(outputs['pred_logits'][idx])
        first_src_logits_sum = torch.sum(src_logits[:, :self.num_known_classes], dim=1)
        
        second_idx = self._get_src_permutation_idx(second_indices)
        second_src_boxes = outputs['pred_boxes'][second_idx]
        second_target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, second_indices)], dim=0)
        
        src_logits = torch.sigmoid(outputs['pred_logits'][second_idx])
        second_src_logits_sum = torch.sum(src_logits[:, :self.num_known_classes], dim=1)

        first_P_G_GIOU = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        
        second_P_G_GIOU = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(second_src_boxes),
            box_ops.box_cxcywh_to_xyxy(second_target_boxes)))
        
        obj_list = []
        iou_list = []
        negative_list = []
        
        for count, iou in enumerate(first_P_G_GIOU):
            if iou > 0.6:
                iou_list.append(iou * 0.8 + 0.2 * first_src_logits_sum[count])
                obj_list.append(outputs['obj'][idx[0][count]][idx[1][count]])
            else:
                negative_list.append(outputs['obj'][idx[0][count]][idx[1][count]])
    
        for count, iou in enumerate(second_P_G_GIOU):
            if iou > 0.6:
                iou_list.append(iou * 0.6 + 0.4 * second_src_logits_sum[count])
                obj_list.append(outputs['obj'][second_idx[0][count]][second_idx[1][count]])
            else:
                negative_list.append(outputs['obj'][idx[0][count]][idx[1][count]])
        
        if len(obj_list) == 0:
            obj_list.append(outputs['obj'][0][0] - outputs['obj'][0][0])
        if len(negative_list) == 0:
            negative_list.append(outputs['obj'][0][0] - outputs['obj'][0][0])
            
        obj_tensor = torch.cat(obj_list, dim=0)
        iou_tensor = torch.tensor(iou_list).to(src_boxes.device)
        negative_tensor = torch.cat(negative_list, dim=0)
        
        negative_gt = (torch.ones(negative_tensor.shape) * 0.5).to(src_boxes.device)

        if len(obj_list) == 0:
            losses = {}
            losses['loss_object'] = obj_tensor
        else:
            obj_loss = F.l1_loss(obj_tensor, iou_tensor, reduction='none')
            neg_loss = F.l1_loss(negative_tensor, negative_gt, reduction='none')
            losses = {}
            losses['loss_object'] = obj_loss.sum() / num_boxes + neg_loss.sum() / num_boxes
        return losses

    def loss_eobj(self, outputs, targets, indices, second_indices, num_boxes):
        assert 'obj' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        src_logits = torch.sigmoid(outputs['pred_logits'][idx])
        first_src_logits_sum = torch.sum(src_logits[:, :self.num_known_classes], dim=1)
        
        second_idx = self._get_src_permutation_idx(second_indices)
        second_src_boxes = outputs['pred_boxes'][second_idx]
        second_target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, second_indices)], dim=0)
        
        src_logits = torch.sigmoid(outputs['pred_logits'][second_idx])
        second_src_logits_sum = torch.sum(src_logits[:, :self.num_known_classes], dim=1)

        first_P_G_GIOU = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        
        second_P_G_GIOU = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(second_src_boxes),
            box_ops.box_cxcywh_to_xyxy(second_target_boxes)))
        
        obj_list = []
        iou_list = []
        negative_list = []
        
        for count, iou in enumerate(first_P_G_GIOU):
            if iou > 0.6:
                iou_list.append(first_src_logits_sum[count])
                obj_list.append(outputs['obj'][idx[0][count]][idx[1][count]])
            else:
                negative_list.append(outputs['obj'][idx[0][count]][idx[1][count]])
    
        for count, iou in enumerate(second_P_G_GIOU):
            if iou > 0.6:
                iou_list.append(second_src_logits_sum[count])
                obj_list.append(outputs['obj'][second_idx[0][count]][second_idx[1][count]])
            else:
                negative_list.append(outputs['obj'][idx[0][count]][idx[1][count]])
        
        if len(obj_list) == 0:
            obj_list.append(outputs['obj'][0][0] - outputs['obj'][0][0])
        if len(negative_list) == 0:
            negative_list.append(outputs['obj'][0][0] - outputs['obj'][0][0])
            
        obj_tensor = torch.cat(obj_list, dim=0)
        iou_tensor = torch.tensor(iou_list).to(src_boxes.device)
        negative_tensor = torch.cat(negative_list, dim=0)
        
        negative_gt = (torch.ones(negative_tensor.shape) * 0.5).to(src_boxes.device)

        if len(obj_list) == 0:
            losses = {}
            losses['loss_eobject'] = obj_tensor
        else:
            obj_loss = F.l1_loss(obj_tensor, iou_tensor, reduction='none')
            neg_loss = F.l1_loss(negative_tensor, negative_gt, reduction='none')
            losses = {}
            losses['loss_eobject'] = obj_loss.sum() / num_boxes + neg_loss.sum() / num_boxes
        return losses
   
       
        
    def loss_masks(self, outputs, targets, indices, tmp_indices,  num_boxes, epoch):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, pos_indices, neg_indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'objs': self.loss_obj,
            'eobjs': self.loss_eobj,
            'masks': self.loss_masks
        }
        
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        
        return loss_map[loss](outputs, targets, pos_indices, neg_indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() 
                              if k != 'aux_outputs' and k != 'enc_outputs'}

        indices, second_indices  = self.matcher(outputs_without_aux, targets)
        
        num_boxes = sum(len(t["labels"]) for t in targets)
        
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, 
                                    device=next(iter(outputs.values())).device)
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == 'eobjs':
                continue
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, 
                                       second_indices, num_boxes, **kwargs))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, second_indices  = self.matcher(aux_outputs, targets)
                
                for loss in self.losses:
                    if loss == 'masks' or loss == 'eobjs':
                        continue
                    
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, 
                                          second_indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'enc_outputs' in outputs: 
            enc_outputs = outputs['enc_outputs']
            
            bin_targets = copy.deepcopy(targets)
            
            indices, second_indices  = self.matcher(enc_outputs, bin_targets)
            
            for loss in self.losses:
                if loss == 'masks' or loss == 'objs':
                    continue
                
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, 
                                      second_indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
     
        
        return losses



class PostProcess(nn.Module):

    def __init__(self, args=None):
        super().__init__()
        self.num_known_classes = getattr(args, 'num_known_classes', 20)
        self.obj_threshold = getattr(args, 'obj_threshold', 0.2)
        self.known_threshold = getattr(args, 'known_threshold', 0.1)
        self.unknown_threshold = getattr(args, 'unknown_threshold', 0.1)
        self.known_score_weight = getattr(args, 'known_score_weight', 2.0)
        self.unknown_score_weight = getattr(args, 'unknown_score_weight', 0.32)
        self.post_topk = getattr(args, 'post_topk', 300)
        self.nms_iou_threshold = getattr(args, 'nms_iou_threshold', 0.7)
        self.known_unknown_nms_iou_threshold = getattr(
            args, 'known_unknown_nms_iou_threshold', None
        )
        if self.known_unknown_nms_iou_threshold is None:
            self.known_unknown_nms_iou_threshold = self.nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        out_objectness = outputs['obj']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_scores = out_objectness.squeeze(-1)
        
        obj_keep_mask = obj_scores > self.obj_threshold
        
        prob = out_logits.sigmoid()
        
        obj_scores_expanded = obj_scores.unsqueeze(-1)
        adjusted_prob = prob * obj_scores_expanded
        
        
        num_classes = prob.shape[-1]
        unknown_class_index = num_classes - 1
        num_known_classes = min(self.num_known_classes, unknown_class_index)
        adjustment_factors = torch.ones(num_classes, device=prob.device)
        
        adjustment_factors[:num_known_classes] = self.known_score_weight
        
        adjustment_factors[unknown_class_index] = self.unknown_score_weight
        
        
        
        known_class_probs = prob[:, :, :num_known_classes]
        
        max_known_class_prob, _ = known_class_probs.max(dim=-1)
        
        unknown_scores = obj_scores * (1 - max_known_class_prob)
        
        adjusted_prob[:, :, unknown_class_index] = unknown_scores

        adjusted_prob = adjusted_prob * adjustment_factors.unsqueeze(0).unsqueeze(0)
        
        batch_size = out_logits.shape[0]
        results = []
        
        for batch_idx in range(batch_size):
            current_obj_keep = obj_keep_mask[batch_idx]
            current_prob = adjusted_prob[batch_idx]
            current_unknown_scores = unknown_scores[batch_idx]
            current_obj_scores = obj_scores[batch_idx]
            current_boxes = out_bbox[batch_idx]
            
            valid_indices = current_obj_keep.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                results.append({
                    'scores': torch.tensor([], device=out_logits.device),
                    'labels': torch.tensor([], device=out_logits.device, dtype=torch.long),
                    'boxes': torch.tensor([], device=out_logits.device).reshape(0, 4)
                })
                continue
            
            valid_prob = current_prob[valid_indices]
            valid_unknown_scores = current_unknown_scores[valid_indices]
            valid_obj_scores = current_obj_scores[valid_indices]
            valid_boxes = current_boxes[valid_indices]
            
            max_scores, max_classes = valid_prob.max(dim=-1)
            
            sorted_indices = max_scores.argsort(descending=True)
            top_k = min(self.post_topk, len(sorted_indices))
            top_indices = sorted_indices[:top_k]
            
            top_scores = max_scores[top_indices]
            top_classes = max_classes[top_indices]
            top_boxes = valid_boxes[top_indices]
            top_obj_scores = valid_obj_scores[top_indices]
            
            top_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(top_boxes)
            
            img_h, img_w = target_sizes[batch_idx].unbind(0)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h])
            top_boxes_scaled = top_boxes_xyxy * scale_fct
            
            keep_indices = batched_nms(
                top_boxes_scaled,
                top_scores,
                top_classes,
                iou_threshold=self.nms_iou_threshold
            )
            
            nms_scores = top_scores[keep_indices]
            nms_classes = top_classes[keep_indices]
            nms_boxes = top_boxes_scaled[keep_indices]
            nms_obj_scores = top_obj_scores[keep_indices]
            
            
            known_mask = nms_classes < num_known_classes
            unknown_mask = nms_classes == unknown_class_index
            
            known_keep = known_mask & (nms_scores > self.known_threshold)
            unknown_keep = unknown_mask & (nms_scores > self.unknown_threshold)
            
            final_keep = known_keep | unknown_keep
            
            final_scores = nms_scores[final_keep]
            final_labels = nms_classes[final_keep]
            final_boxes = nms_boxes[final_keep]

            final_known_mask = final_labels < num_known_classes
            final_unknown_mask = final_labels == unknown_class_index
            if final_known_mask.any() and final_unknown_mask.any():
                unknown_boxes = final_boxes[final_unknown_mask]
                known_boxes = final_boxes[final_known_mask]
                ious, _ = box_ops.box_iou(unknown_boxes, known_boxes)
                suppress_unknown = ious.max(dim=1).values > self.known_unknown_nms_iou_threshold

                keep_after_cross_nms = torch.ones_like(final_labels, dtype=torch.bool)
                unknown_positions = final_unknown_mask.nonzero(as_tuple=True)[0]
                keep_after_cross_nms[unknown_positions[suppress_unknown]] = False

                final_scores = final_scores[keep_after_cross_nms]
                final_labels = final_labels[keep_after_cross_nms]
                final_boxes = final_boxes[keep_after_cross_nms]
            
            results.append({
                'scores': final_scores,
                'labels': final_labels,
                'boxes': final_boxes
            })

        return results


class PostProcess1(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        out_objectness = outputs['obj']
      
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        
        obj_scores = out_objectness.max(-1).values
        
        obj_threshold = 0.3
        
        obj_keep_mask = obj_scores > obj_threshold
        
        prob = out_logits.sigmoid()
        
        max_class_scores, predicted_classes = prob.max(dim=-1)
        
        
        class_threshold = 0.2
        
        is_known = max_class_scores > class_threshold
        
        
        sum_class_scores = prob[:, :, :-1].sum(dim=-1)
        
        complete_scores = obj_scores * 0.6 + sum_class_scores * 0.4
        
        batch_size = out_logits.shape[0]
        results = []
        
        for batch_idx in range(batch_size):
            current_obj_keep = obj_keep_mask[batch_idx]
            
            current_is_known = is_known[batch_idx]
            
            current_predicted_classes = predicted_classes[batch_idx]
            
            current_scores = complete_scores[batch_idx]
            
            current_boxes = out_bbox[batch_idx]
            
            valid_indices = current_obj_keep.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                results.append({
                    'scores': torch.tensor([], device=out_logits.device),
                    'labels': torch.tensor([], device=out_logits.device, dtype=torch.long),
                    'boxes': torch.tensor([], device=out_logits.device).reshape(0, 4)
                })
                continue
            
            valid_scores = current_scores[valid_indices]
            valid_is_known = current_is_known[valid_indices]
            valid_predicted_classes = current_predicted_classes[valid_indices]
            valid_boxes = current_boxes[valid_indices]
            
            
            
            valid_labels = torch.where(
                valid_is_known,
                valid_predicted_classes,
                torch.tensor(80, device=out_logits.device)
            )
            
            
            sorted_indices = valid_scores.argsort(descending=True)
            
            top_k = min(300, len(sorted_indices))
            top_indices = sorted_indices[:top_k]
            
            final_scores = valid_scores[top_indices]
            final_labels = valid_labels[top_indices]
            final_boxes = valid_boxes[top_indices]
            
            final_boxes = box_ops.box_cxcywh_to_xyxy(final_boxes)
            
            img_h, img_w = target_sizes[batch_idx].unbind(0)
            
            scale_fct = torch.stack([img_w, img_h, img_w, img_h])
            
            final_boxes = final_boxes * scale_fct
            
            results.append({
                'scores': final_scores,
                'labels': final_labels,
                'boxes': final_boxes
            })

        return results

          
class NMSPostProcess(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        print('use nms process')
        
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs, n_queries, n_cls = out_logits.shape

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
        
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
        
        all_boxes = all_indexes // out_logits.shape[2]
        all_labels = all_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1,1,4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for b in range(bs):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]

            if n_queries * n_cls > 10000:
                pre_topk = score.topk(10000).indices
                box = box[pre_topk]
                score = score[pre_topk]
                lbls = lbls[pre_topk]

            keep_inds = batched_nms(box, score, lbls, 0.7)[:100]
            
            results.append({
                'scores': score[keep_inds],
                'labels': lbls[keep_inds],
                'boxes':  box[keep_inds],
            })

        return results



class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        h = [hidden_dim] * (num_layers - 1)
        
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


def build(args):
    num_classes = args.num_classes
  
    if args.dataset == "coco_panoptic":
        num_classes = 250
    num_known_classes = min(getattr(args, 'num_known_classes', num_classes - 1),
                            max(num_classes - 1, 1))
    
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    
    
    
    
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    matcher = build_matcher(args)
    
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_object': args.obj_loss_coef
    }

    weight_dict['loss_giou'] = args.giou_loss_coef
    
    weight_dict['loss_eobject'] = args.obj_loss_coef
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    
    if args.aux_loss:
        aux_weight_dict = {}
        
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'objs', 'eobjs']
    
    if args.masks:
        losses += ["masks"]
    
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses,
                            focal_alpha=args.focal_alpha,
                            num_known_classes=num_known_classes)
    
    criterion.to(device)
    
    postprocessors = {'bbox': PostProcess(args)}
    
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        
        if args.dataset == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
