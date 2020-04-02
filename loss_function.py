import torch
import numpy as np

class NtXent(torch.nn.Module):
    # TODO prüfen ob korrekt
    def __init__(self, processor, batch_size, ):
        super(NtXent, self).__init__()
        self.batch_size = batch_size
        self.processor = processor
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self.get_mask().type(torch.bool)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def get_mask(self):
        '''
        Mask:
        true false false ... false true false ... false
        false true false ... false false true ... false
        false false true ... false false false ... false
        ...   ...   ...  ...  ...   ...  ...   ...  ...
        true false false ... false false false ... true
        false true false ... false false false ... false
        false false true ... false false false ... false
        ...   ...   ...  ...  ...   ...  ...   ...  ...
        false false false ... false true false ... true

        1x diagonal: from (0,0) to (511, 511)
        1x diagonal: from (255,0) to (511, 255)
        1x diagonal: from (0,255) to (255, 511)
        '''
        mask_length = 2 * self.batch_size
        mask = torch.from_numpy((np.eye(mask_length) + np.eye(mask_length, k=-self.batch_size) + np.eye(mask_length, k=self.batch_size)))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.processor)

    def forward(self, zi, zj):
        temperature = 0.5
        representations = torch.cat([zj, zi], dim=0) # Concatenate zj and zi
        similarity_matrix = self.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0))

        positives = self.filter_positive_samples(similarity_matrix)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= temperature
        labels = torch.zeros(2 * self.batch_size).to(self.processor).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

    def filter_positive_samples(self, similarity_matrix):
        upper_positives = torch.diag(similarity_matrix, self.batch_size)
        lower_positives = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([upper_positives, lower_positives]).view(2 * self.batch_size, 1)
        return positives
