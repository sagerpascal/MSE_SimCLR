import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models


# Doku PyTorch: https://pytorch.org/docs/stable/nn.html
class ResNetModel(nn.Module):
    '''
    Use a resnet18 model as base encoder, and add the projection head (consisting of two
    additional layer) sequentially to this network
    '''
    def __init__(self, out_dim=512):
        super(ResNetModel, self).__init__()
        # initialize resnet with 18 layers
        resnet_18 = models.resnet18(pretrained=False)
        number_of_features = resnet_18.fc.in_features
        self.base_encoder = nn.Sequential(*list(resnet_18.children())[:-1])  # remove last fully-connected layer

        # sequentially connect the projection head to the base encoder
        self.l1 = nn.Linear(number_of_features, number_of_features)
        self.l2 = nn.Linear(number_of_features, out_dim)

    def forward(self, x):
        '''
        Defines the computation performed at every call.

        extract the feature vector from an augmented view and map this vector to the space where the loss
        function is applied

        :param x: Input data
        :return: extracted feature-vectors h, mapped representations vectors z
        '''
        h = self.base_encoder(x)
        h = h.squeeze()  # Returns a tensor with all the dimensions of input of size 1 removed.
        z = self.l1(h)
        z = functional.relu(z)
        z = self.l2(z)
        return h, z
