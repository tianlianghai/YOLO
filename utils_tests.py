from utils import intersection_over_union, none_max_suppression, mean_average_precision
import torch

def test_intersection_over_union():
    shape = (16, 7, 7, 4)
    boxes_preds = torch.rand(*shape)
    boxes_labels = torch.rand(*shape)

    boxes_same = boxes_labels
    iou = intersection_over_union(boxes_same, boxes_labels)
    return iou

def test_none_max_suppression():
    preds = [
        [0, 0.9, 0, 0, 1, 1],
        [0, 0.8, 0, 0, .9, .9],
        [1, 0.3, 0, 0, 1, 1],
        [1, 0.6, 0, 0, 1, 1]
    ]

    return none_max_suppression(preds)

def test_mean_average_precision():
    # preds : [[train_idx, class, confidence, x1, y1, x2, y2], ...]
    preds = [
        [0, 2, 0.9, 0.2, 0, 1, .9],
        [0, 2, 0.8, 0.2, 0, 1, .9]
    ]
    labels = [
        [0, 2, 0.9, 0.2, 0, 1, .9],
        [0, 1, 0.9, 0.2, 0, 1, .9]

    ]

    return mean_average_precision(preds, labels, num_classes=3)

def main():
    ans = test_mean_average_precision()
    print(ans)
main()