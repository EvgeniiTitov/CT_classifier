import os
import typing as t
from csv import reader, writer
import random

import pydicom.dataset
import pydicom
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from classifier.logger import get_logger
from classifier.config import Config


__all__ = (
    "save_validation_image_paths",
    "create_new_run_folder",
    "visualise_training_results",
    "save_model",
    "get_training_device",
    "get_loss_function",
    "get_optimizer",
    "get_parameters_to_train",
    "reshape_models_head",
    "show_tensor",
    "open_dcm_file",
    "read_annotations",
    "show_dicom_file",
    "get_dcm_file_paths",
    "get_image_paths_and_labels",
    "show_images",
    "save_config",
    "show_np_array"
)


logger = get_logger("utils")


def visualise_training_results(
    acc_history: list[float],
    loss_history: list[float],
    run_folder: str,
    show_plot: bool = False,
) -> None:
    plt.subplot(1, 2, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(acc_history, linewidth=3)

    plt.subplot(1, 2, 2)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(loss_history, linewidth=3)

    if show_plot:
        plt.show()
    plt.savefig(os.path.join(run_folder, "plot.png"))
    logger.info(f"Training graph saved to {run_folder}")


def save_config(
    config: object, run_folder: str, filename: str = "config.txt"
) -> None:
    filepath = os.path.join(run_folder, filename)
    with open(filepath, "w") as file:
        for attr_name, value in config.__dict__.items():
            if not attr_name.startswith("__"):
                file.write(f"{attr_name}: {value}\n")
    logger.info(f"Config saved to {run_folder}")


def save_validation_image_paths(
    valid_paths: list[str], run_folder: str, filename: str = "valid_paths.txt"
) -> None:
    if not os.path.exists(run_folder):
        raise FileNotFoundError(f"Run folder {run_folder} does not exist")
    filepath = os.path.join(run_folder, filename)
    with open(filepath, "w") as file:
        for valid_image_path in valid_paths:
            file.write(valid_image_path + "\n")
    logger.info(f"Validation image paths saved to {run_folder}")


def save_model(model, model_name: str, run_folder: str) -> None:
    if not os.path.exists(run_folder):
        raise FileNotFoundError(f"Run folder {run_folder} does not exist")
    torch.save(model, os.path.join(run_folder, model_name))
    logger.info(f"Model saved to {run_folder}")


def create_new_run_folder(root: str = "../runs") -> str:
    current_runs = os.listdir(root)
    if not len(current_runs):
        run_folder_path = os.path.join(root, "run1")
        os.mkdir(run_folder_path)
    else:
        current_runs.sort(key=lambda item: int(item[-1]))
        latest_run = int(current_runs[-1][-1])
        run_folder_path = os.path.join(root, f"run{latest_run + 1}")
        os.mkdir(run_folder_path)
    return run_folder_path


def get_training_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_loss_function(lf_name: str) -> nn.Module:
    if lf_name == "CROSS_ENTROPY":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Requested loss function is not supported")
    return loss_function


def get_optimizer(
    optimizer_name: str, trainable_params: list
) -> optim.Optimizer:
    if optimizer_name == "ADAM":
        optimizer = optim.Adam(
            params=trainable_params, lr=Config.LR, betas=Config.BETAS
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            params=trainable_params, lr=Config.LR, momentum=Config.MOMENTUM
        )
    else:
        raise NotImplementedError("Requested optimizer is not supported")
    return optimizer


def reshape_models_head(model, number_of_classes: int):
    """Reshapes the last dense layer(s) of the model to the number of classes
    the model will be trained for
    """
    if model.__class__.__name__ == "ResNet":
        number_of_filters = model.fc.in_features
        model.fc = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "AlexNet":
        # 6th Dense layer's input size: 4096
        number_of_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "VGG":
        # For both VGGs 16-19 classifiers are the same
        number_of_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "DenseNet":
        number_of_filters = model.classifier.in_features
        model.classifier = nn.Linear(number_of_filters, number_of_classes)

    elif model.__class__.__name__ == "SqueezeNet":
        model.classifier[1] = nn.Conv2d(
            512, number_of_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.num_classes = number_of_classes
    return model


def get_parameters_to_train(model, fine_tuning: bool) -> list:
    """Returns a list of trainable parameters - the ones for which the
    gradients will be calculated during backprop
    """
    trainable_parameters = model.parameters()
    if fine_tuning:
        return trainable_parameters
    else:
        trainable_parameters = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                trainable_parameters.append(parameter)
        return trainable_parameters


def show_tensor(tensor: torch.Tensor) -> None:
    array = tensor.numpy().transpose(1, 2, 0)
    cv2.imshow("", array)
    cv2.waitKey(0)


def open_dcm_file(filepath: str) -> np.ndarray:
    image_file = pydicom.dcmread(filepath)
    # show_dicom_file(image_file)
    image_arr = np.array(image_file.pixel_array, dtype=np.uint8)
    image_arr = np.repeat(image_arr[..., np.newaxis], 3, axis=2)
    return image_arr


def read_annotations(path: str) -> t.Iterator[list[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} doesn't exist")

    with open(path, mode="r") as file:
        csv_reader = reader(file)
        header = next(csv_reader)
        if header:
            for row in csv_reader:
                yield row


def show_np_array(image: np.ndarray, window_name: str = "") -> None:
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def show_dicom_file(image: pydicom.dataset.FileDataset) -> None:
    image = image.pixel_array
    cv2.imshow("", image)
    cv2.waitKey(0)


def get_dcm_file_paths(folder: str) -> t.Iterator[str]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} doesn't exist")

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for file in os.listdir(subfolder_path):
            if file.lower().endswith(".dcm"):
                yield os.path.join(subfolder_path, file)


def separate_annotations_into_two_classes(
    filepath: str, positive_filepath: str, negative_filepath: str
) -> None:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} doesn't exist")

    with open(filepath, mode="r") as source_file, open(
        positive_filepath, mode="w"
    ) as positive_file, open(negative_filepath, mode="w") as negative_file:

        source_reader = reader(source_file)
        positive_writer = writer(positive_file)
        negative_writer = writer(negative_file)

        header = next(source_reader)
        positive_writer.writerow(header)
        negative_writer.writerow(header)

        for row in source_reader:
            _, _, has_disease, *_ = row
            if has_disease == "1":
                positive_writer.writerow(row)
            else:
                negative_writer.writerow(row)
    logger.info("Done separating the annotations")


def split_train_val(
    image_source_folder: str,
    image_destination: str,
    annotations_file: str,
    class_name: str,
    ratio: float,
    shuffle: bool = True,
) -> None:
    def _move_files(
        rows: list[str],
        source: str,
        destination: str,
        is_training: bool,
        class_name: str,
    ) -> None:
        for row in rows:
            row = eval(row)
            slice_id, case_id, *_ = row  # type: ignore
            file_name = slice_id + ".dcm"  # type: ignore
            source_path = os.path.join(
                source, case_id, file_name  # type: ignore
            )
            destination_path = os.path.join(
                destination,
                "train" if is_training else "val",
                class_name,
                file_name,
            )
            os.rename(source_path, destination_path)

    if not os.path.exists(image_destination):
        os.mkdir(image_destination)

    with open(annotations_file, mode="r") as file:
        csv_reader = reader(file)
        _ = next(csv_reader)
        lines = [str(line) for line in csv_reader]
        logger.info(f"Read {len(lines)} lines")

        if shuffle:
            random.shuffle(lines)

        total_images = len(lines)
        val_images = int(total_images * ratio)
        train_images = int(total_images * (1 - ratio))
        logger.info(f"Val images: {val_images}; Train images: {train_images}")

        val_subset = lines[:val_images]
        train_subset = lines[val_images:]

        _move_files(
            val_subset,
            image_source_folder,
            image_destination,
            is_training=False,
            class_name=class_name,
        )
        logger.info("Moved validation images")
        _move_files(
            train_subset,
            image_source_folder,
            image_destination,
            is_training=True,
            class_name=class_name,
        )
        logger.info("Moved training images")


def get_image_paths_and_labels() -> tuple[list[str], list[int]]:
    image_paths, labels = [], []
    for row in read_annotations("../data/training/annotations.csv"):
        slice_id, case_id, label, *_ = row
        image_path = os.path.join(
            "../data/training", case_id, slice_id + ".dcm"
        )
        image_paths.append(image_path)
        labels.append(int(label))

    return image_paths, labels


def show_images(
    images: list[np.ndarray],
    window_name: str = "",
    horizontal: bool = True,
) -> None:
    try:
        arr = np.concatenate(images, axis=1 if horizontal else 0)
    except Exception as e:
        logger.error("Failed while concatenating images into a single array")
        raise e
    cv2.imshow(window_name, arr)
    cv2.waitKey(0)


if __name__ == "__main__":
    # separate_annotations_into_two_classes(
    #     filepath="../data/training/annotations.csv",
    #     positive_filepath="../data/training/positives.csv",
    #     negative_filepath="../data/training/negatives.csv",
    # )

    # split_train_val(
    #     image_source_folder="../data/training",
    #     image_destination="../data/image_folder_format",
    #     annotations_file="../data/training/negatives.csv",
    #     class_name="negative",
    #     ratio=0.25,
    #     shuffle=True,
    # )

    # image_paths, labels = get_image_paths_and_labels()
    # for i, (image_path, label) in enumerate(zip(image_paths, labels)):
    #     if i > 10:
    #         break
    #     print(image_path, label)

    # image = open_dcm_file("../data/training/CID_0081375920/ID_1938ae4a7.dcm")
    # print(image.shape)

    # print(_create_new_run_folder())

    # image = open_dcm_file("../data/training/CID_0081375920/ID_1938ae4a7.dcm")
    # print(image.shape)
    #
    # cv2.imshow("", image)
    # cv2.waitKey(0)
    #
    # from torchvision.transforms.functional import to_pil_image
    #
    # pil_image = to_pil_image(image)
    # print("Done")

    print(get_training_device())

    # save_config(Config)
