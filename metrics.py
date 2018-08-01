import torch


def iou(prediction, target, reduce=True):
    batch_size = prediction.size(0)
    prediction = prediction.view(batch_size, -1)
    target = target.view(batch_size, -1)

    intersection = torch.sum(target & prediction, dim=1)
    union = torch.sum(target | prediction, dim=1)

    union[union == 0] = 1

    # Handle no target object
    target_sum = torch.sum(target, dim=1) == 0
    prediction_sum = torch.sum(prediction, dim=1) == 0
    no_object = target_sum & prediction_sum

    intersection[no_object] = 1
    union[no_object] = 1

    if reduce:
        return torch.mean(intersection.float() / union.float())

    return intersection.float() / union.float()


def mean_iou(precision, target, classes=None, reduce=True):
    if classes is None:
        classes = [1]

    ious = []
    for cls in classes:
        prediction_for_cls = precision.clone()
        prediction_for_cls[prediction_for_cls != cls] = 0
        prediction_for_cls[prediction_for_cls == cls] = 1
        target_for_cls = target.clone()
        target_for_cls[target_for_cls != cls] = 0
        target_for_cls[target_for_cls == cls] = 1

        iou_for_cls = iou(prediction_for_cls.byte(), target_for_cls.byte(), reduce)
        ious.append(iou_for_cls)

    ious = torch.stack(ious, dim=0)
    ious = torch.mean(ious, dim=0)

    return ious


def true_positives(prediction, target, reduce=True):
    batch_size = prediction.size(0)

    positives = target == 1
    true_positive_locations = (positives + prediction) == 2
    true_positive_score = torch.sum(true_positive_locations.view(batch_size, -1))

    if reduce:
        return torch.mean(true_positive_score.float())

    return true_positive_score


def mAP(prediction, target, thresholds=None, reduce=True):
        if thresholds is None:
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        iou_scores = []
        for t in thresholds:
            iou_score = mean_iou(prediction, target, reduce=False)
            iou_scores.append(iou_score > t)

        iou_scores = torch.stack(iou_scores, dim=1)
        mAP_scores = torch.mean(iou_scores.float(), dim=1)

        if reduce:
            return torch.mean(mAP_scores)

        return mAP_scores