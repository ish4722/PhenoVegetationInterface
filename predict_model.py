from config import *
from backend.segmentation.train_predict import predict
from backend.preprocessing.dataset_loader import Dataset, Dataloader
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Initialize dataset and dataloader for predictions
    print("Initializing dataset...")
    dataset = Dataset(DATASET_PATH, n_classes=N_CLASSES)
    dataset_indexes = list(range(len(dataset)))
    _, test_indexes = train_test_split(dataset_indexes, test_size=0.15, random_state=42)
    test_dataloader = Dataloader(test_indexes, dataset, batch_size=BATCH_SIZE)

    # Load the pre-trained model and predict
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    predictions = predict(MODEL_SAVE_PATH, test_dataloader)

    print("Prediction complete.")
    # Add any visualization or evaluation code here
