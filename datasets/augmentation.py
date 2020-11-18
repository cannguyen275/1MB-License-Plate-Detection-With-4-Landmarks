import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import torch.nn.functional as F
import skimage.transform
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import albumentations as A
from imgaug import augmenters as iaa
import random


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            # iaa.Sometimes(0.25, iaa.GammaContrast(gamma=(0, 1.75))),
            # iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.3))),
            # iaa.Sometimes(0.25, iaa.pillike.Autocontrast(cutoff=(0, 15.0))),
            # iaa.Grayscale(alpha=(0.0, 1.0)),
            # iaa.Sometimes(0.15, iaa.MotionBlur(k=5, angle=[-45, 45])),
            iaa.Sometimes(0.35,
                          iaa.OneOf([iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                                     iaa.GammaContrast(gamma=(0, 1.75)),
                                     iaa.pillike.Autocontrast(cutoff=(0, 15.0)),
                                     iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.3)))
                                     ])),
            # iaa.Fliplr(0.5),
            # iaa.Sometimes(0.35,
            #               iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
            #                          iaa.Dropout2d(p=0.5),
            #                          iaa.CoarseDropout(0.1, size_percent=0.5),
            #                          iaa.SaltAndPepper(0.1),
            #                          ])),
            # iaa.Sometimes(0.15,
            #               iaa.OneOf([
            #                   iaa.Clouds(),
            #                   iaa.Fog(),
            #                   iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
            #                   iaa.Rain(speed=(0.1, 0.3))
            #               ])),
            # iaa.Sometimes(0.5, iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)))
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugment_Albumentations(object):
    """
    """

    def __init__(self, width=512, height=512):
        super(ImgAugment_Albumentations, self).__init__()
        self.width = width
        self.height = height
        # Set data augmentation name, check its github  for more data augmentation methods.
        self.transform_train = A.Compose(
            [
                # A.PadIfNeeded(min_height=1080, min_width=1980, border_mode=1, p=0.6),
                # A.IAACropAndPad(percent=(-0.8, 0.8), p=0.6),
                # A.OneOf([A.RandomScale(scale_limit=(0.005, 0.5), p=0.3),
                #          A.RandomScale(scale_limit=(0.1, 1), p=0.2),
                #          A.RandomScale(scale_limit=(1, 3), p=0.3)]),

                # A.RandomResizedCrop(height, width),
                A.Rotate(limit=60, p=0.3),
                # A.IAAAffine(p=0.3),
                # A.IAAPerspective(p=0.5),
                # A.OneOf([
                #     A.IAAAdditiveGaussianNoise(p=0.5),
                #     A.GaussNoise(p=0.6),
                # ]),
                A.Resize(height, width)
            ],
            bbox_params=A.BboxParams(format='pascal_voc', min_area=256, min_visibility=0.5,
                                     label_fields=["bbox_classes"]),
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=["keypoints_classes"])
        )
        self.augment_imgaug = ImgAugTransform()

    def __call__(self, image, bbs, kps, bbox_classes, keypoints_classes):
        """
        Where the magic happened.
        :param image: input image. Can be opencv image
        :param bbs: the bounding box of objects.
        :param kps: the keypoints of objects
        """
        img_aug = self.augment_imgaug(image)
        img = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
        transformed = self.transform_train(image=img, bboxes=bbs, bbox_classes=bbox_classes, keypoints=kps,
                                           keypoints_classes=keypoints_classes)
        for index, (x, y) in enumerate(transformed['keypoints']):
            x = max(x, 0)
            y = max(y, 0)
            x = min(x, self.width)
            y = min(y, self.height)
            transformed['keypoints'][index] = (x, y)
        return transformed['image'], transformed['bboxes'], transformed['keypoints']
