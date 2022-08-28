import typing as t
import abc

import numpy as np
import torchvision
import torch
from PIL import Image

from classifier.logger import get_logger


__all__ = ["TrainedBrainClassificationModel"]


logger = get_logger("trained_model")


class BaseTrainedModel(abc.ABC):
    @abc.abstractmethod
    def predict_batch(self, images: list[np.ndarray]) -> list[t.Any]:
        ...


class TrainedBrainClassificationModel(BaseTrainedModel):

    def __init__(
        self,
        model_weights_path: str,
        model_classes: list[str],
        preprocessing_pipeline: torchvision.transforms.Compose,
        inference_device: t.Literal["CPU", "GPU"] = "CPU"
    ) -> None:
        self._model = self._load_model(model_weights_path)

        if inference_device == "GPU":
            if torch.cuda.is_available():
                self._device = torch.device("cuda:0")
                logger.info("Cuda available, inference will be done on GPU")
            else:
                self._device = torch.device("cpu")
                logger.info("Cuda unavailable, inference will be done on CPU")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)

        self._classes = model_classes
        self._preprocessing_pipeline = preprocessing_pipeline
        logger.info("Model initialized for inference")

    def predict_batch(self, images: list[np.ndarray]) -> list[t.Any]:
        if not len(images):
            return []
        preprocessed_batch = []
        for image in images:
            preprocessed_image = self._preprocessing_pipeline(
                Image.fromarray(image)
            )
            preprocessed_batch.append(
                torch.unsqueeze(preprocessed_image, 0)
            )
        torch_batch = torch.cat(preprocessed_batch)

        try:
            torch_batch = torch_batch.to(self._device)
        except Exception as e:
            logger.error("Failed while moving batch to device. Out of memory?")
            raise e

        with torch.no_grad():
            output = self._model(torch_batch)

        return [
            self._classes[out.data.numpy().argmax()] for out in output.cpu()
        ]

    @staticmethod
    def _load_model(weights_path: str):
        try:
            model = torch.load(weights_path)
        except Exception as e:
            logger.error(f"Failed to load the model from {weights_path}")
            raise e
        model.eval()
        return model

    def __call__(self, images: list[np.ndarray]) -> list[t.Any]:
        return self.predict_batch(images)
