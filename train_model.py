from config import *
from backend.preprocessing.annotation_to_mask import generate_masks_from_annotations
from backend.segmentation.train_predict import train_model

if __name__ == "__main__":
    # Generate masks from JSON annotations
    print("Generating masks...")
    generate_masks_from_annotations(DATASET_PATH)

    # Train the model
    print("Training the model...")
    train_model(DATASET_PATH, MODEL_SAVE_PATH, N_CLASSES, BACKBONE, LEARNING_RATE, EPOCHS, BATCH_SIZE)

    print(f"Model training complete. Saved model to {MODEL_SAVE_PATH}.")
