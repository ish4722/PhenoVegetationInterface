import os
from datetime import datetime

def apply_time_filter(image_dir, start_time, end_time):
    """Filter images in a directory based on timestamps."""
    filtered_images = []
    for file_name in os.listdir(image_dir):
        try:
            # Extract time from the file name
            file_time = datetime.strptime(file_name[4:], "%Y_%m_%d_%H%M%S.jpg")
            if start_time <= file_time <= end_time:
                filtered_images.append(os.path.join(image_dir, file_name))
        except ValueError:
            continue
    return filtered_images
