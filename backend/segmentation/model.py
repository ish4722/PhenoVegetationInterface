from keras.models import load_model
import cv2
import numpy as np

def load_efficientnet_model(model_path):
    """Load the pre-trained EfficientNet model."""
    return load_model(model_path, compile=False)

def predict_masks(model, image):
    """Predict masks for coniferous and deciduous trees."""
    resized_image = cv2.resize(image, (256, 256))
    input_image = np.expand_dims(resized_image, axis=0)
    return model.predict(input_image, verbose=0) > 0.5
