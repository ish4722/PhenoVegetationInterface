from datetime import datetime

# Configurations for dataset paths
# Paths
DATASET_PATH = "/Users/ishan/Downloads/DAI_Project/labelled"
MODEL_SAVE_PATH = "saved_model/my_model.h5"

# Model Parameters
BACKBONE = "efficientnetb3"
LEARNING_RATE = 0.001
EPOCHS = 40
BATCH_SIZE = 1
N_CLASSES = 3  # 3 classes: background, deciduous, coniferous

# Configuration file
IMAGE_DIR = "path/to/images"
MODEL_PATH = "/Users/ishan/Downloads/my_model-2.h5"
OUTPUT_EXCEL = "output.xlsx"

START_TIME = datetime(2019, 9, 5, 18, 30)
END_TIME = datetime(2019, 9, 7, 19, 0)

