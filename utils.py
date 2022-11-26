import torch
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels):
    """
    The boxes should be represent as cornors
    Parameter:
        boxes_preds: shape(num_boxes, S, S, 4)
    """

    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]

    box2_x1 = boxes_labels[..., 0:1]    
    box2_y1 = boxes_labels[..., 1:2]    
    box2_x2 = boxes_labels[..., 2:3]    
    box2_y2 = boxes_labels[..., 3:4]    


    # torch.return_types.max(
    # values=tensor([ 8,  9, 10, 11]),
    # indices=tensor([2, 2, 2, 2]))
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)


    # .clamp(0) is for the case when they don't intersect, namely, those two values 
    # would both be negative, and we will get a positive intersection, but that's not right
    # we shold get a zero.
    # and from perspective of data, the initial data is random, so that box1_x1 can be bigger
    # than box1_x2, which it should not be, and in this case the intersection will be zero too,
    # and in this case even pred is the same as label, the iou will be 0, however, in practice
    # the label's coordinates should be normal. So we should not see this scenario.
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # could this be a negative? don't know, if so, we should add an abs(), for now,
    # let's just keep it simple.
    # when implementing the test function, I realize that the output boxes from early training
    # iteration will just be some random values, and the area can be negative. So let's add abs()
    # and the intersection has been clamped at 0, so we don't need that.
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    union = box1_area + box2_area - intersection

    # Aladdin add a small scale to the denominator, for numerical stability, but I don't see why,
    # because intersection should always be smaller than union.
    # Let's just keep this part simple and see what happens.
    return intersection / union


def none_max_suppression(predictions, prob_threshold=0.4, iou_threshold=0.5):
    
    # predictinos: [[2, 0.9, x1, y1, x2, y2], [], []]
    assert type(predictions) == list

    bboxes = [box for box in predictions if box[1] > prob_threshold]

    # I got this from Aladin's video, but this line of code menas that a box
    # is responsible for that object if it has the highest confidence,
    # this is not completely corrospond to the paper saying that has the highest iou.

    # After a while, I realize that this NMS part is not the place where 'responsible' is 
    # calculated. NMS is only used for cleaning purpose.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    nms_bboxes = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(box[2:]),
                torch.tensor(chosen_box[2:])
            ) < iou_threshold
        ]
        nms_bboxes.append(chosen_box)

    return nms_bboxes


def mean_average_precision(bboxes_pred, bboxes_labels, iou_threshold=0.5, num_classes=20):
    """
    VOC mAP calculation, for iou_threshold always equal to 0.5
    bboxes_pred(list): [[train_idx, class, confidence, x1, y1, x2, y2], ...] 
    """

    bboxes_pred = sorted(bboxes_pred, key=lambda x:x[1], reverse=True)
    average_precisions = []

    for c in range(num_classes):

        # Filter classes 
        detections = []
        ground_truths = []

        for box in bboxes_pred:
            if box[1] == c:
                detections.append(box)

        for box in bboxes_labels:
            if box[1] == c:
                ground_truths.append(box)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_ground_truths = len(ground_truths)

        # sort by confidence
        detections = sorted(detections, key=lambda x:x[2], reverse=True)

        # so if the 0th image has three bboxes in class 0, then this will be {0:3, ...}
        true_boxes_count = Counter([gth[0] for gth in ground_truths])

        # {0:3, ...} -> {0:[0, 0, 0], ...}
        for key, value in true_boxes_count.items():
            true_boxes_count[key] = torch.zeros(value)

        for detection_idx, detection in enumerate(detections):
            best_iou = 0
            best_idx = 0

            ground_truths_of_image = [gth for gth in ground_truths if gth[0]==detection[0]]
            
            # compare one detection with all the groundtruth in that particular image.
            # only select those ground truth with same train index.
            for gth_idx, ground_truth in  enumerate(ground_truths_of_image):
                    
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), 
                    torch.tensor(ground_truth[3:])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_idx = gth_idx
            
            if best_iou > iou_threshold:
                # this ground truth has not been detected.
                if true_boxes_count[detection[0]][best_idx] == 0:
                    true_boxes_count[detection[0]][best_idx] = 1
                    TP[detection_idx] = 1
                
                else:
                    FP[detection_idx] = 1
            
            else:
                FP[detection_idx] = 1

        # in class for loop, out of detections for loop
        TP_cumsum = TP.cumsum(dim=0)
        FP_cumsum = FP.cumsum(dim=0)

        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        recalls = TP_cumsum / total_ground_truths
        # add the (0, 1) point into the P-R curve
        precisions = torch.cat([torch.Tensor([1]), precisions])
        recalls = torch.cat([torch.Tensor([0]), recalls])

        AP = torch.trapezoid(precisions, recalls)
        average_precisions.append(AP)
    
    return sum(average_precisions) / len(average_precisions)
