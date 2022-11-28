import torch
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    The boxes should be represent as cornors
    Parameter:
        boxes_preds: shape(num_boxes, S, S, 4)
    """
    if box_format == "corner":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]    
        box2_y1 = boxes_labels[..., 1:2]    
        box2_x2 = boxes_labels[..., 2:3]    
        box2_y2 = boxes_labels[..., 3:4]   
    
    elif box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2


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


def none_max_suppression(predictions, prob_threshold=0.4, iou_threshold=0.5, box_format="midpoint"):
    
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
    epsilon = 1e-6

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
        
        if total_ground_truths == 0:
            continue

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

        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = TP_cumsum / (total_ground_truths + epsilon)
        # add the (0, 1) point into the P-R curve
        precisions = torch.cat([torch.Tensor([1]), precisions])
        recalls = torch.cat([torch.Tensor([0]), recalls])

        AP = torch.trapezoid(precisions, recalls)
        average_precisions.append(AP)
    
    return sum(average_precisions) / len(average_precisions)



def get_bboxes(data_loader, model, iou_threshold=0.5, threshold=0.4, box_format="midpoint", device="cuda"):
        
    all_pred_boxes = []
    all_true_boxes = []
    model.eval()
    train_idx = 0
        
    for batch_idx, (x, labels) in enumerate(data_loader):
        x, labels = x.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(x)
        batch_size = x.shape[0]
        
        true_boxes = cellboxes_to_boxes(labels)
        # (batch, 7, 7, 30) -> (batch, 49, 30)
        bboxes = cellboxes_to_boxes(preds)
        
        for idx in range(batch_size):
            nms_boxes = none_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold= threshold,
                box_format = box_format
            )
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            
            for box in true_boxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            
            train_idx +=1
    
    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7, num_classes=20, num_boxes=2):
    #(batch, 7, 7, 30) -> (batch, 7, 7, 6)
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, num_classes + num_boxes * 5)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[...,25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1-best_box) + bboxes2 * best_box
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1/S * (best_boxes[..., :1] + cell_indices)
    y = 1/S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1,3))
    w_h = 1/S *(best_boxes[..., 2: 4])
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., :num_classes].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., num_classes], predictions[..., num_classes]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    
    return converted_preds
    
def cellboxes_to_boxes(out, S=7):
    # (batch, 7 * 7 * 30) -> [[[class, confi, x1, x2, w, h],... for 49 times],... for batch times]
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S *S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []
    
    for img_idx in range(out.shape[0]):
        
        bboxes = []
        
        for bbox_idx in range(S *S):
            # why not use tolist()?
            # add a bbox from a grid of a picture into the image bbox list
            bboxes.append([x.item() for x  in  converted_pred[img_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)
    return all_bboxes
        
        
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
