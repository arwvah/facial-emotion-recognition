# SPDX-License-Identifier: AGPL-3.0-only
"""Whole-face transforms and trainer for the optional Ultralytics experiment."""

from copy import copy

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class ResizeFace:
    """Match the application's OpenCV linear resize exactly, without cropping."""

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        return Image.fromarray(
            cv2.resize(np.asarray(image), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        )


class FaceDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix="", names=None):
        super().__init__(root, args, augment, prefix, names)
        steps = [ResizeFace(args.imgsz)]
        if augment:
            steps += [
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            ]
        # Ultralytics classifiers use RGB [0, 1], without ImageNet mean/std normalization.
        self.torch_transforms = T.Compose([*steps, T.ToTensor()])


class FaceValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return FaceDataset(img_path, self.args, False, self.args.split, self.names)

    def get_stats(self):
        stats = super().get_stats()
        # Eight-class top-5 is too lenient for selecting an expression classifier.
        stats["fitness"] = stats["metrics/accuracy_top1"]
        return stats


class FaceTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return FaceDataset(img_path, self.args, mode == "train", mode, self.data["names"])

    def get_validator(self):
        return FaceValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
