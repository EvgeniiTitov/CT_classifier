import typing as t

from classifier.trained_model import TrainedBrainClassificationModel
from classifier.utils import read_annotations, open_dcm_file
from classifier.image_transforms import get_transformations
from classifier.logger import get_logger


"""
Overcomplicated this bit but I ran out of time and patience.
Could have been incorporated into the training step

STDOUT:
calculate_metrics 47: Loaded expected results
trained_model 38: Cuda available, inference will be done on GPU
trained_model 48: Model initialized for inference
calculate_metrics 57: Calculating the metrics
calculate_metrics 62: Processed 0 images
calculate_metrics 62: Processed 100 images
calculate_metrics 62: Processed 200 images
calculate_metrics 62: Processed 300 images
calculate_metrics 62: Processed 400 images
calculate_metrics 62: Processed 500 images
calculate_metrics 62: Processed 600 images
calculate_metrics 62: Processed 700 images
calculate_metrics 62: Processed 800 images
calculate_metrics 62: Processed 900 images
calculate_metrics 62: Processed 1000 images
calculate_metrics 62: Processed 1100 images
calculate_metrics 62: Processed 1200 images
calculate_metrics 62: Processed 1300 images
calculate_metrics 62: Processed 1400 images
calculate_metrics 62: Processed 1500 images
calculate_metrics 62: Processed 1600 images
calculate_metrics 62: Processed 1700 images
calculate_metrics 62: Processed 1800 images
calculate_metrics 62: Processed 1900 images

calculate_metrics 85: TP: 156; TN: 1566; FP: 104; FN: 115
calculate_metrics 87: Sensitivity: 0.5756; Specificity: 0.9377
"""


# TODO: Run inference in batch


logger = get_logger("calculate_metrics")


MODEL_WEIGHTS = "runs/run1/weights.pth"
VALID_IMAGES_LIST = "runs/run1/valid_paths.txt"


def load_expected_results() -> t.Mapping[str, str]:
    mapping = {}
    for row in read_annotations("data/training/annotations.csv"):
        slice_id, case_id, label, *_ = row
        mapping[f"{case_id}-{slice_id}.dcm"] = str(label)
    return mapping


def get_validation_image_path() -> t.Iterator[str]:
    with open(VALID_IMAGES_LIST, "r") as file:
        for line in file:
            yield line.strip()


def calculate_metrics(
    tp: int, tn: int, fp: int, fn: int
) -> tuple[float, float]:
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def main():
    expected_results = load_expected_results()
    logger.info("Loaded expected results")

    preprocessing = get_transformations(augmentation_required=False)["valid"]
    model = TrainedBrainClassificationModel(
        model_weights_path=MODEL_WEIGHTS,
        model_classes=["0", "1"],
        preprocessing_pipeline=preprocessing,
        inference_device="GPU",
    )

    logger.info("Calculating the metrics")
    TP, TN = 0, 0
    FP, FN = 0, 0
    for i, path in enumerate(get_validation_image_path()):
        if not i % 100:
            logger.info(f"Processed {i} images")

        image = open_dcm_file(path[3:])
        actual_result = model.predict_batch([image])[0]

        # I hate windows with its endless problems (\\ vs \ vs /)
        case_id = path[17:31]
        slice_id = path[32:]
        key = f"{case_id}-{slice_id}"

        expected_result = expected_results[key]
        # print(expected_result, actual_result)
        if expected_result == actual_result:
            if expected_result == "1":
                TP += 1
            else:
                TN += 1
        else:
            if expected_result == "1" and actual_result == "0":
                FN += 1
            else:
                FP += 1

    logger.info(f"TP: {TP}; TN: {TN}; FP: {FP}; FN: {FN}")
    sensitivity, specificity = calculate_metrics(TP, TN, FP, FN)
    logger.info(
        f"Sensitivity: {sensitivity:.4f}; Specificity: {specificity:.4f}"
    )


if __name__ == "__main__":
    main()
