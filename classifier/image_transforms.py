import typing as t

from torchvision import transforms

from classifier.config import Config


__all__ = ["get_transformations"]


Transforms = t.MutableMapping[str, transforms.Compose]


def get_transformations(augmentation_required: bool) -> Transforms:
    if augmentation_required:
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(
                        (Config.INPUT_IMAGE_SIZE, Config.INPUT_IMAGE_SIZE)
                    ),
                    transforms.RandomRotation(degrees=Config.ROTATION_DEGREES),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize(
                        (Config.INPUT_IMAGE_SIZE, Config.INPUT_IMAGE_SIZE)
                    ),
                    transforms.CenterCrop(Config.INPUT_IMAGE_SIZE),
                    transforms.ToTensor(),
                ]
            ),
        }
    else:
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(
                        (Config.INPUT_IMAGE_SIZE, Config.INPUT_IMAGE_SIZE)
                    ),
                    transforms.ToTensor(),
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize(
                        (Config.INPUT_IMAGE_SIZE, Config.INPUT_IMAGE_SIZE)
                    ),
                    transforms.CenterCrop(Config.INPUT_IMAGE_SIZE),
                    transforms.ToTensor(),
                ]
            ),
        }
    return data_transforms
