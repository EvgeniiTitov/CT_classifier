import argparse

from classifier.trained_model import TrainedBrainClassificationModel
from classifier.image_transforms import get_transformations
from classifier.utils import open_dcm_file, show_np_array


"""
Script for inference.

Example:
python inference.py
--model_weights runs/run1/weights.pth
--image_path data/training/CID_4a8dbcc8eb/ID_bfa3c4c43.dcm
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    preprocessing = get_transformations(augmentation_required=False)["valid"]
    model = TrainedBrainClassificationModel(
        model_weights_path=args.model_weights,
        model_classes=["no_disease", "has_disease"],
        preprocessing_pipeline=preprocessing,
        inference_device="GPU",
    )
    image = open_dcm_file(args.image_path)
    result = model([image])
    show_np_array(image, f"Result: {result}")


if __name__ == "__main__":
    main()
