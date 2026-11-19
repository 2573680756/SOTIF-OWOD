
import numpy
import copy
def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.0
        for t in numpy.arange(0.0, 1.1, 0.1):
            if numpy.sum(rec >= t) == 0:
                p = 0
            else:
                p = numpy.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = numpy.concatenate(([0.], rec, [1.]))
        mpre = numpy.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = numpy.maximum(mpre[i - 1], mpre[i])

        i = numpy.where(mrec[1:] != mrec[:-1])[0]

        ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_evaluate(detections, annotations, cid, ovthresh=0.5, use_07_metric=True):


    class_gt_bboxes = {}
    npos = 0
    for image_name in annotations:
        R = [obj[:4] for obj in annotations[image_name]]
        bbox = numpy.array(R)
        difficult = numpy.array([0] * len(R)).astype(numpy.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_gt_bboxes[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    det_image_files = []
    confidences = []
    det_bboxes = []
    class_dict = detections
    for image_file in class_dict:
        
        dets = class_dict[image_file]
        for k in range(dets.shape[0]):
            det_image_files.append(image_file)
            det_bboxes.append(dets[k, 0:4])
            confidences.append(dets[k, -1])
    det_bboxes = numpy.array(det_bboxes)
    confidences = numpy.array(confidences)

    num_dets = len(det_image_files)
    tp = numpy.zeros(num_dets)
    fp = numpy.zeros(num_dets)
   
    if det_bboxes.shape[0] == 0:
        return 0., 0., 0.

    sorted_ind = numpy.argsort(-confidences)
    det_bboxes = det_bboxes[sorted_ind, :]
    det_image_files = [det_image_files[x] for x in sorted_ind]

    for d in range(num_dets):
        R = class_gt_bboxes[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -numpy.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ixmin = numpy.maximum(BBGT[:, 0], bb[0])
            iymin = numpy.maximum(BBGT[:, 1], bb[1])
            ixmax = numpy.minimum(BBGT[:, 2], bb[2])
            iymax = numpy.minimum(BBGT[:, 3], bb[3])
            iw = numpy.maximum(ixmax - ixmin + 1., 0.)
            ih = numpy.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = numpy.max(overlaps)
            jmax = numpy.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    state = [copy.deepcopy(tp), copy.deepcopy(fp)]
    stp = sum(tp)
    recall = stp / npos
    precision = stp / (stp + sum(fp))

    fp = numpy.cumsum(fp)
    tp = numpy.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / numpy.maximum(tp + fp, numpy.finfo(numpy.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return recall, precision, ap, rec, prec, state, det_image_files