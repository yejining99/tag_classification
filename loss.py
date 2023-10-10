import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, distance, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, text_embedding, keyword_embedding, label):
        distance = self.distance(text_embedding, keyword_embedding)
        loss = torch.mean((1-label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss
    
def euclidean_distance(x, y):
    return torch.norm(x-y, dim=1, p=2)
    
def loss_and_distance(loss_name, distance):
    ### distance function
    if distance == "pairwise_distance":
        distance = F.pairwise_distance
    elif distance == "cosine_similarity":
        distance = F.cosine_similarity
    elif distance == "euclidean_distance":
        distance = euclidean_distance
    else:
        raise ValueError("distance must be one of 'pairwise_distance', 'cosine_similarity', 'euclidean_distance'")
    
    ### loss function
    if loss_name == "contrastive":
        return ContrastiveLoss(distance), distance
    else:
        raise ValueError("loss_name must be one of 'contrastive'")
    
