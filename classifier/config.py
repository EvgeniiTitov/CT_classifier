class Config:
    # LOGGER
    LOGGER_VERBOSE = True
    LOGGER_FORMAT = (
        "%(asctime)s %(name)s %(levelname)s %(lineno)s: %(message)s"
    )

    # TRAINING
    ROTATION_DEGREES = 45
    INPUT_IMAGE_SIZE = 224
    BATCH_SIZE = 14
    AUG_REQUIRED = False
    TEST_SIZE = 0.25
    SHUFFLING = True
    PRETRAINED = True
    FINE_TUNING = False
    OPTIMIZER = "ADAM"
    LOSS_FUNCTION = "CROSS_ENTROPY"
    LR = 0.001
    BETAS = (0.9, 0.999)
    MOMENTUM = 0.9
    SCHEDULER = True
    SCHEDULER_STEP = 5
    SCHEDULER_GAMMA = 0.1
    EPOCHS = 1
