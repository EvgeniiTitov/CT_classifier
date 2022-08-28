import torch.utils.data
import torchvision.models
from sklearn.model_selection import train_test_split
import copy

from torchvision import models
import torch.optim as optim

from classifier.utils import (
    get_image_paths_and_labels,
    reshape_models_head,
    get_parameters_to_train,
    get_optimizer,
    get_loss_function,
    get_training_device,
    save_model,
    visualise_training_results,
    create_new_run_folder,
    save_validation_image_paths,
)
from classifier.logger import get_logger
from classifier.config import Config
from classifier.image_transforms import get_transformations
from classifier.custom_data_loader import BrainScanDataset


logger = get_logger("training")


def main():
    run_folder_path = create_new_run_folder()
    logger.info(
        f"New run folder created. Results could be found in {run_folder_path}"
    )

    image_paths, labels = get_image_paths_and_labels()
    logger.info(f"Got {len(image_paths)} images and {len(labels)} labels")

    unique_labels = set(labels)
    logger.info(f"Training will be done for classes: {unique_labels}")

    (
        train_images,
        valid_images,
        train_labels,
        valid_labels,
    ) = train_test_split(
        image_paths,
        labels,
        test_size=Config.TEST_SIZE,
        random_state=1,
        shuffle=Config.SHUFFLING,
    )
    logger.info(
        f"Training done split into train and valid. "
        f"Training images {len(train_images)}; "
        f"Valid images: {len(valid_images)}"
    )
    save_validation_image_paths(valid_images, run_folder_path)

    image_transforms = get_transformations(Config.AUG_REQUIRED)
    logger.info(
        f"Image transformations obtained. Augmentation: {Config.AUG_REQUIRED}"
    )

    train_dataset = BrainScanDataset(
        image_paths=train_images,
        labels=train_labels,
        image_transformations=image_transforms["train"],
    )
    valid_dataset = BrainScanDataset(
        image_paths=valid_images,
        labels=valid_labels,
        image_transformations=image_transforms["valid"],
    )
    logger.info("Custom datasets initialized")

    dataset_sizes = {"train": len(train_images), "valid": len(valid_images)}

    data_loaders = {
        "train": torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=Config.BATCH_SIZE
        ),
        "valid": torch.utils.data.DataLoader(
            valid_dataset, batch_size=Config.BATCH_SIZE
        ),
    }
    logger.info("Dataloaders initialized")

    weights = (
        torchvision.models.ResNet18_Weights.DEFAULT
        if Config.PRETRAINED
        else None
    )
    model = models.resnet18(weights=weights)
    logger.info(f"Pretrained weights used: {Config.PRETRAINED}")

    if Config.FINE_TUNING:
        logger.info("Fine tuning - all model parameters are training")
    else:
        logger.info("No fine tuning - freezing all layers")
        for parameter in model.parameters():
            parameter.requires_grad = False

    total_classes = len(unique_labels)
    if total_classes != 1000:
        model = reshape_models_head(model, total_classes)
        logger.info(
            f"Model head replaced with a new one for {total_classes} classes"
        )

    trainable_params = get_parameters_to_train(
        model, fine_tuning=Config.FINE_TUNING
    )
    logger.info("Got the list of trainable parameters")

    optimizer = get_optimizer(Config.OPTIMIZER, trainable_params)
    logger.info(f"Optimizer {Config.OPTIMIZER} initialized")

    loss_function = get_loss_function(Config.LOSS_FUNCTION)
    logger.info("Loss function initialized")

    scheduler = None
    if Config.SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=Config.SCHEDULER_STEP,
            gamma=Config.SCHEDULER_GAMMA,
        )
        logger.info("Scheduler initialized and will be used during training")
    else:
        logger.info("Scheduler won't be used during training")

    device = get_training_device()
    logger.info(f"Device for training: {device}")
    model.to(device)

    logger.info("Training commences:")
    val_accuracy_history, val_loss_history = [], []
    best_val_accuracy = 0
    best_acc_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, Config.EPOCHS + 1):
        logger.info(f"Epoch {epoch} / {Config.EPOCHS}")

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for batch, labels in data_loaders[phase]:
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    activations = model(batch)
                    loss = loss_function(activations, labels)
                    _, class_preds = torch.max(activations, dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * batch.size(0)
                running_corrects += torch.sum(class_preds == labels.data)

            if phase == "train" and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]
            logger.info(
                f"{phase.upper()} Loss: {epoch_loss:.4f} "
                f"Accuracy: {epoch_accuracy:.4f}"
            )

            if phase == "valid":
                val_accuracy_history.append(epoch_accuracy.item())
                val_loss_history.append(epoch_loss)

            # TODO: Prioritise loss instead
            if phase == "valid" and epoch_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                best_acc_epoch = epoch

    logger.info(
        f"Training complete. Best accuracy {best_val_accuracy:.4f} achieved on"
        f" the epoch {best_acc_epoch}"
    )

    model.load_state_dict(best_model_weights)
    logger.info("Loaded best model weights")

    save_model(model, model_name="weights.pth", run_folder=run_folder_path)
    visualise_training_results(
        val_accuracy_history, val_loss_history, run_folder_path
    )


if __name__ == "__main__":
    main()
