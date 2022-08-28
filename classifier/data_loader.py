import typing as t
import os

import torch
from torchvision import datasets, transforms

from classifier.logger import get_logger
from classifier.config import Config


"""
This bit is unnecessary and could be ignored.
I wanted to follow the ImageFolder format with the dataset but Torch doesn't
support the .dcm format
"""


__all__ = ["DatasetLoader"]


logger = get_logger("data_loader")


class DatasetLoader:
    def __init__(
        self,
        path_to_dataset: str,
        augmentation_required: bool,
        input_image_size: int,
        batch_size: int,
    ) -> None:
        self._dataset_path = path_to_dataset
        self._aug_required = augmentation_required
        self._image_size = input_image_size
        self._batch_size = batch_size

    def get_datasets(self) -> tuple:
        data_transforms = self._get_transformations()
        logger.info("Image transformations obtained")

        image_datasets = {
            phase: datasets.ImageFolder(
                root=os.path.join(self._dataset_path, phase),
                transform=data_transforms[phase],
            )
            for phase in ["train", "val"]
        }
        logger.info("Image datasets created for both phases")

        data_loaders = {
            phase: torch.utils.data.DataLoader(
                dataset=image_datasets[phase],
                batch_size=self._batch_size,
                shuffle=True,
            )
            for phase in ["train", "val"]
        }
        logger.info("Data loaders created")

        dataset_sizes = {
            phase: len(image_datasets[phase]) for phase in ["train", "val"]
        }
        class_names = image_datasets["train"].classes
        logger.info(
            f"Training classes: {class_names}; "
            f"Dataset sizes: {dataset_sizes}"
        )
        return image_datasets, data_loaders, dataset_sizes, class_names

    def _get_transformations(self) -> t.Mapping[str, transforms.Compose]:
        if self._aug_required:
            logger.info("Augmentation will be applied")
            data_transformations = {
                "train": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._image_size, self._image_size)
                        ),
                        transforms.RandomRotation(
                            degrees=Config.ROTATION_DEGREES
                        ),
                        transforms.ColorJitter(),
                        transforms.ToTensor(),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._image_size, self._image_size)
                        ),
                        transforms.CenterCrop(self._image_size),
                        transforms.ToTensor(),
                    ]
                ),
            }
        else:
            data_transformations = {
                "train": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._image_size, self._image_size)
                        ),
                        transforms.ToTensor(),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(
                            (self._image_size, self._image_size)
                        ),
                        transforms.CenterCrop(self._image_size),
                        transforms.ToTensor(),
                    ]
                ),
            }
        return data_transformations
