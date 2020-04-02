import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

np.random.seed(0)


class ImagesDataSet:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.input_shape = (96, 96, 3)

    def get_data_loaders(self):
        # Download data and apply transformation two times on each image
        train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True, transform=DataTransformation(self.get_data_augmentation()))
        train_loader, valid_loader = self.init_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_data_augmentation(self):
        color_distortion = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_distortion], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(self.input_shape[0]),
                                              transforms.ToTensor()])
        return data_transforms

    def init_data_loaders(self, train_dataset):
        validation_set_size = 0.05
        split = int(np.floor(validation_set_size * len(train_dataset)))

        # List from 0..len(train_dataset) -> random shuffled
        index = list(range(len(train_dataset)))
        np.random.shuffle(index)
        training_data_index, validation_data_index = index[split:], index[:split]

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(training_data_index), num_workers=0, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(validation_data_index), num_workers=0, drop_last=True)
        return train_loader, valid_loader


class DataTransformation:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class GaussianBlur:
    def __init__(self, image_size):
        self.window_size = int(image_size * 0.1)

    def __call__(self, data):
        min_ = 0.1
        max_ = 2.0
        data = np.array(data)
        if np.random.random_sample() < 0.5:
            sigma = (max_ - min_) * np.random.random_sample() + min_
            data = cv2.GaussianBlur(data, (self.window_size, self.window_size), sigma)
        return data
