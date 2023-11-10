import torch
from torchvision import transforms as transform_lib
import cv2
from PIL import Image, ImageOps
import numpy as np

class BYOLDataTransform():
    def __init__(
        self,
        crop_size,
        mean,
        std,
        blur_prob=(1.0, 0.1),
        solarize_prob=(0.0, 0.2),
        GaussianBlur=0,
        ):
        assert len(blur_prob) == 2 and len(solarize_prob) == 2, 'atm only 2 views are supported'
        self.GaussianBlur = GaussianBlur
        self.crop_size = crop_size
        self.normalize = transform_lib.Normalize(mean=mean, std=std)
        self.color_jitter = transform_lib.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.transforms = [self.build_transform(bp, sp) for bp, sp in zip(blur_prob, solarize_prob)]

    def build_transform(self, blur_prob, solarize_prob):
        
        if self.GaussianBlur == 0:
            print('not use GaussianBlur')
            transforms = transform_lib.Compose([
                # transform_lib.Resize((256, 256)),
                transform_lib.RandomResizedCrop(self.crop_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        else:
            print('use GaussianBlur')
            transforms = transform_lib.Compose([
                # transform_lib.Resize((256, 256)),
                transform_lib.RandomResizedCrop(self.crop_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        return transforms

    def __call__(self, x):
        return [t(x) for t in self.transforms]

class BYOLDataTransform_amlp():
    def __init__(
        self,
        crop_size,
        mean,
        std,
        blur_prob=(1.0, 0.1),
        solarize_prob=(0.0, 0.2),
        aug_type=0,
        data_aug_crop=0.8,
        ):
        assert len(blur_prob) == 2 and len(solarize_prob) == 2, 'atm only 2 views are supported'
        self.aug_type = aug_type
        self.crop_size = crop_size
        self.normalize = transform_lib.Normalize(mean=mean, std=std)
        self.color_jitter = transform_lib.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.data_aug_crop = data_aug_crop
        self.transforms = [self.build_transform(bp, sp) for bp, sp in zip(blur_prob, solarize_prob)]

    def build_transform(self, blur_prob, solarize_prob):
        
        if self.aug_type == 0:
            transforms = transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.crop_size, scale=(self.data_aug_crop, 1)),
                transform_lib.RandomHorizontalFlip(),
                # transform_lib.RandomGrayscale(p=0.2),
                # transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        if self.aug_type == 1:
            transforms = transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.crop_size, scale=(self.data_aug_crop, 1)),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                # transform_lib.RandomGrayscale(p=0.2),
                # transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                # transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        if self.aug_type == 2:
            transforms = transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.crop_size, scale=(self.data_aug_crop, 1)),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                # transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                # transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        if self.aug_type == 3:
            transforms = transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.crop_size, scale=(self.data_aug_crop, 1)),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                # transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        if self.aug_type == 4:
            transforms = transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.crop_size, scale=(self.data_aug_crop, 1)),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        print('---------------------------------------')
        print('transforms', transforms)
        print('---------------------------------------')
        return transforms

    def __call__(self, x):
        return [t(x) for t in self.transforms]


class MOCODataTransform():

    def __init__(
        self,
        crop_size,
        mean,
        std,
        blur_prob=(0.5, 0.5),
        solarize_prob=(0.0, 0.2),
        GaussianBlur=0,
        ):
        assert len(blur_prob) == 2 and len(solarize_prob) == 2, 'atm only 2 views are supported'
        self.GaussianBlur = GaussianBlur
        self.crop_size = crop_size
        self.normalize = transform_lib.Normalize(mean=mean, std=std)
        self.color_jitter = transform_lib.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.transforms = [self.build_transform(bp, sp) for bp, sp in zip(blur_prob, solarize_prob)]

        print('use MOCODataTransform ')

    def build_transform(self, blur_prob, solarize_prob):
        
        if self.GaussianBlur == 0:
            print('not use GaussianBlur')
            transforms = transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.crop_size, scale=(0.2, 1)),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                transform_lib.ToTensor(),
                self.normalize
            ])
        else:
            print('use GaussianBlur')
            transforms = transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.crop_size,scale=(0.2, 1)),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                transform_lib.ToTensor(),
                self.normalize
            ])
        return transforms

    def __call__(self, x):
        return [t(x) for t in self.transforms]


class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))


class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)

if __name__ =="__main__":

    train_transform = BYOLDataTransform(
        crop_size=32,
        mean=(0.491, 0.482, 0.447),
        std=(0.247, 0.243, 0.261),
        blur_prob=[.0, .0],
        solarize_prob=[.0, .2],
        GaussianBlur=1,
        )

    x = torch.randn(36, 3, 224, 224)
    y = train_transform(x)
    print(len(y), y[0].shape, y[1].shape)
