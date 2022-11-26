import torch
from torch import nn
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
    def __init__(self, lambda_coord, lambda_noobj, S, C, B):
        super.__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = S
        self.C = C
        self.B = B
        
    def forward(self, preds, targets):
        # pred: the output from darknet, 
        # shape is (batch, S * S * C + B * 5)
        # we should resize this
        S = self.S
        C = self.C
        B = self.B
 
        preds = preds.reshape(-1, S, S, C + B*5)
        
    
  
        