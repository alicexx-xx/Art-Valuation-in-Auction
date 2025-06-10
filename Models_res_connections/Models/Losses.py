import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, b_lower, b_upper, tail_weight=4, middle_weight=3):
        super().__init__()
        self.tail_weight = tail_weight
        self.middle_weight = middle_weight
        self.b_lower = b_lower
        self.b_upper = b_upper
    
    def forward(self, y_pred, y_true):
        weights = torch.where((y_true < self.b_lower)|(y_true > self.b_upper), 
                              torch.tensor(self.tail_weight, device=y_true.device),
                              torch.tensor(self.middle_weight, device=y_true.device))
        squared_errors = (y_pred - y_true)**2
        weighted_squared_errors = weights * squared_errors

        return weighted_squared_errors.mean()
    
class MeanQuarticError(nn.Module):
    def __init__(self, b_lower, b_upper, tail_weight=4, middle_weight=3):
        super().__init__()
        self.b_lower = b_lower
        self.b_upper = b_upper
    
    def forward(self, y_pred, y_true):
        return ((y_pred - y_true)**4).mean()