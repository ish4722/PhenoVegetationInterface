import cv2
import numpy as np

def detect_blurred_phenocam(image):
    """Detect blurred images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    return laplacian_var < 1000

def categorize_darken(image):
    """Detect darkened images."""
    height = image.shape[0]
    bottom_third = image[int(height * (2/3)):, :]
    gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    return mean_intensity < 30 and std_intensity < 15

def detect_snow(image):
    """Detect snowy images."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_snow = np.array([0, 0, 180], dtype=np.uint8)
    upper_snow = np.array([180, 25, 255], dtype=np.uint8)
    snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)
    snowy_percentage = np.sum(snow_mask == 255) / snow_mask.size
    return snowy_percentage > 0.0053
