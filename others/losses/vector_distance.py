from others.losses import CustomLoss

import torch.nn.functional as F


class CosineSimilarity(CustomLoss):
    def __init__(self, negative=True):
        super(CosineSimilarity, self).__init__()
        self.negative = negative

    def forward(self, x, y):
        cosine_similarity = F.cosine_similarity(x.flatten(start_dim=1), y.flatten(start_dim=1), dim=1).mean()
        return 1 - cosine_similarity if self.negative else cosine_similarity
