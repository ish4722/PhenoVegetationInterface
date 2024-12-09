import os
import keras
import segmentation_models as sm
from tensorflow.keras.models import load_model

def train_model(dataset_path, model_save_path, n_classes, backbone, lr, epochs, batch_size):
    """Train the model and save it."""
    from sklearn.model_selection import train_test_split
    from backend.preprocessing.dataset_loader import Dataset, Dataloader
    from backend.preprocessing.dataset_loader import aug

    # Load dataset and create dataloaders
    dataset = Dataset(dataset_path, n_classes, augmentation=aug())
    dataset_indexes = list(range(len(dataset)))
    train_indexes, test_indexes = train_test_split(dataset_indexes, test_size=0.15, random_state=42)
    train_dataloader = Dataloader(train_indexes, dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = Dataloader(test_indexes, dataset, batch_size=batch_size)

    # Define model, loss, optimizer
    sm.set_framework("tf.keras")
    model = sm.Unet(backbone, classes=n_classes, activation="softmax")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=sm.losses.CategoricalFocalLoss() + sm.losses.JaccardLoss(),
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
    )

    # Train the model
    model.fit(
        train_dataloader,
        validation_data=test_dataloader,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(),
            keras.callbacks.EarlyStopping("val_loss", patience=5),
        ],
    )
    model.save(model_save_path)

def predict_images(model_path, test_dataloader):
    """Load the model and perform predictions."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    model = load_model(model_path, compile=False)
    predictions = model.predict(test_dataloader)
    return predictions
