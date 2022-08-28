import typing as t

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from classifier.exceptions import NumberOfImagesDoesntMatchLabelsError
from classifier.utils import open_dcm_file


__all__ = ["BrainScanDataset"]


class BrainScanDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        image_transformations: t.Optional[transforms.Compose],
    ) -> None:
        if len(image_paths) != len(labels):
            raise NumberOfImagesDoesntMatchLabelsError()

        self._image_paths = image_paths
        self._labels = labels
        self._transforms = image_transformations
        self._number_images = len(image_paths)

    def __len__(self) -> int:
        return self._number_images

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self._image_paths[index]
        image = open_dcm_file(image_path)
        image = to_pil_image(image)
        label = self._labels[index]
        if self._transforms:
            image = self._transforms(image)
        return image, label
