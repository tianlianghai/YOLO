import torch
from torch import nn
from utils import intersection_over_union
class YoloLoss(nn.Module):
    """
    Calculate the yolo loss with 4 parts in paper
    Paramters:
        preds (tensor): shape of (batch_size, S, S, C + B*5)
        targets: same shape as preds, except the class probability is 
        either 0 or 1, rather than a decimal in preds, and the C is either 0 or 1,
        note that this C is different from the confidence score in the paper, it's now 
        purely Pr(Obj), 
    """
    def __init__(self, lambda_coord=5, lambda_noobj=0.5, S=7, C=20, B=2):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = S
        self.C = C
        self.B = B
        self.mse = nn.MSELoss(reduction="sum")
        
    def forward(self, preds, targets):
        # pred: the output from darknet, 
        # shape is (batch, S * S * C + B * 5)
        # we should resize this
        S = self.S
        C = self.C
        B = self.B
 
        preds = preds.reshape(-1, S, S, C + B*5)

        box_target = targets[..., C+1: C+5]
        # responsible
        iou1 = intersection_over_union(preds[..., C+1: C+5], box_target).unsqueeze(0)
        iou2 = intersection_over_union(preds[..., C+6: C+10], box_target).unsqueeze(0)
        best_box = torch.argmax(torch.cat((iou1, iou2), dim=0), dim=0)
        
        # ious = intersection_over_union(preds[..., C+1:C+5], box_target).unsqueeze(0)
        # for b in range(1, B):
        #     iou = intersection_over_union(preds[..., C + 5 * b + 1 : C + 5 * (b+1)], box_target).unsqueeze(0)
        #     ious = torch.cat((ious, iou), dim=0)
        # best_box = torch.argmax(ious, dim=0) # j

        exist = targets[..., C:C+1]  # Iobj_i

        # Loss of Coordinates
        # this implemetation only works for 2 boxes, for general cases, we should rewrite this part
        box_responsible = exist * (
            (1 - best_box) * preds[..., C+1:C+5]
            + best_box * preds[..., C+6:C+10]
        )
        box_truth = exist * box_target

        # this 1e-6 solved nan problem in model parameters
        box_responsible[..., 2:] =torch.sign(box_responsible[..., 2:]) * torch.sqrt(torch.abs(box_responsible[...,2:] + 1e-6))
        box_truth[..., 2:] = torch.sqrt(box_truth[..., 2:])
        
        loss_coords = self.mse(
            torch.flatten(box_responsible, start_dim=0, end_dim=-2),
            torch.flatten(box_truth, start_dim=0, end_dim=-2)
        )

        # Loss of object probability
        probability_pred =  (
            (1 - best_box) * preds[..., C:C+1]
            + best_box * preds[..., C+5:C+6]
        )
        probability_target = targets[..., C:C+1]
        
        loss_obj = self.mse(
            torch.flatten(exist * probability_pred),
            torch.flatten(exist * probability_target)
        )


        # Loss of noobj, let's say all of the boxes in a cell is 'responsible' for predict that 
        # there is no object in that cell, and this thought comes from aladdin@youtube. 
        
        # writing in this way is totally wrong, and this caused low mean AP problem
        # loss_noobj = self.mse(
        #     torch.flatten(1 - preds[..., C:C+1]),
        #     torch.flatten(1 - targets[..., C:C+1])
        # )

        # the start dim maybe not necessary, let's just keep it anyway, just accord to 
        # aladdin's video.
        loss_noobj = self.mse(
            torch.flatten((1- exist) * preds[..., C:C+1] , start_dim=1),
            torch.flatten((1- exist) * targets[..., C:C+1], start_dim=1)
        )
        
        loss_noobj += self.mse(
            torch.flatten((1- exist) * preds[..., C+5:C+6], start_dim=1),
            torch.flatten((1- exist) * targets[..., C:C+1], start_dim=1)
        )
        

        # Loss of class
        loss_class = self.mse(
            torch.flatten(exist * preds[...,:C], end_dim=-2),
            torch.flatten(exist * targets[..., :C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * loss_coords
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_class
        )

        return loss

 
  
        