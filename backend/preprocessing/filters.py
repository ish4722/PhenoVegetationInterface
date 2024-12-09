import cv2
import numpy as np

def apply_filters(filtered_images, filters):
    """
    Apply user-specified filters (blurry, darkened, snowy) to a list of images.

    Args:
        filtered_images (list): List of image file paths to filter.
        filters (list): List of filters to apply (e.g., ['blurry', 'darkened', 'snowy']).

    Returns:
        list: Filtered list of image file paths.
    """
    def is_blurry(image):
        """Detect blurred images."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        return laplacian_var < 1000

    def is_darkened(image):
        """Detect darkened images."""
        height = image.shape[0]
        bottom_third = image[int(height * (2 / 3)):, :]
        gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        return mean_intensity < 30 and std_intensity < 15

    def is_snowy(image):
        """Detect snowy images."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_snow = np.array([0, 0, 180], dtype=np.uint8)
        upper_snow = np.array([180, 25, 255], dtype=np.uint8)
        snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)
        snowy_percentage = np.sum(snow_mask == 255) / snow_mask.size
        return snowy_percentage > 0.0053

    # Filter the images based on user-specified filters
    retained_images = []
    for image_path in filtered_images:
        image = cv2.imread(image_path)
        remove = False

        if 'blurry' in filters and is_blurry(image):
            print(f"Filtered out: {image_path} (blurry)")
            remove = True

        if 'darkened' in filters and is_darkened(image):
            print(f"Filtered out: {image_path} (darkened)")
            remove = True

        if 'snowy' in filters and is_snowy(image):
            print(f"Filtered out: {image_path} (snowy)")
            remove = True

        if not remove:
            retained_images.append(image_path)

    return retained_images
