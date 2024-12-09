from config import *
from backend.preprocessing.time_filter import apply_time_filter
from backend.preprocessing.filters import detect_blurred_phenocam, detect_snow
from backend.segmentation.model import load_efficientnet_model, predict_masks
from backend.segmentation.clustering import perform_clustering
from backend.analysis.vegetation_indices import calculate_vegetation_indices
from backend.analysis.excel_writer import save_to_excel
import cv2

def main():
    # Step 1: Filter images
    filtered_images = apply_time_filter(IMAGE_DIR, START_TIME, END_TIME)
    
    # Step 2: Apply filters
    for image_path in filtered_images:
        image = cv2.imread(image_path)
        if detect_blurred_phenocam(image):
            print(f"{image_path} is blurred.") 
        if detect_snow(image):
            print(f"{image_path} is snowy.")

    # Step 3: Predict masks
    model = load_efficientnet_model(MODEL_PATH)
    for image_path in filtered_images:
        image = cv2.imread(image_path)
        masks = predict_masks(model, image)

        # Step 4: Perform clustering and ROI extraction
        points = extract_points_from_masks(masks)  # Assume you extract points here
        centers, labels = perform_clustering(points, roi_area=20000, n_clusters=5)

        # Step 5: Calculate vegetation indices
        indices = calculate_vegetation_indices(red=100, green=150, blue=50)  # Example values
        save_to_excel(indices, OUTPUT_EXCEL)

if __name__ == "__main__":
    main()
